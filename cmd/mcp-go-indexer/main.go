package main

import (
	"bytes"
	"context"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/bytedance/sonic"
	"github.com/google/uuid"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"github.com/qdrant/go-client/qdrant"
	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/golang"
)

const (
	// Update this to match qwen3-embedding's exact output dimension (e.g., 3584, 4096, etc.)
	VectorDimension = 4096 // https://huggingface.co/Qwen/Qwen3-Embedding-8B
	CollectionName  = "codebase_index"
)

// Global variables for our configuration
var (
	ollamaURL      string
	embeddingModel string
	qClient        *qdrant.Client
)

func main() {
	ctx := context.Background()

	// read
	// 1. Read Environment Variables injected by the MCP client config
	ollamaURL = getEnvOrDefault("OLLAMA_URL", "http://localhost:11434")
	embeddingModel = getEnvOrDefault("EMBEDDING_MODEL", "qwen3-embedding:8b")

	qdrantHost := getEnvOrDefault("QDRANT_HOST", "localhost")
	qdrantPortStr := getEnvOrDefault("QDRANT_PORT", "6334")
	qdrantPort, err := strconv.Atoi(qdrantPortStr)
	if err != nil {
		log.Fatalf("Invalid QDRANT_PORT: %v", err)
	}

	// 1. Initialize Qdrant gRPC Client (Port 6334 is default for gRPC)
	qClient, err = qdrant.NewClient(&qdrant.Config{
		Host:   qdrantHost,
		Port:   qdrantPort,
		UseTLS: false,
	})
	if err != nil {
		log.Fatalf("Failed to connect to Qdrant: %v", err)
	}
	defer func() {
		_ = qClient.Close()
	}()

	// 2. Ensure the collection exists
	ensureCollection(ctx, qClient)

	// 3. Initialize MCP Server
	s := server.NewMCPServer("go-codebase-indexer", "1.0.0")

	// --- Tool 1: Index Project ---
	indexTool := mcp.NewTool("index_project",
		mcp.WithDescription("Scans the project, parses Go files, and indexes functions into Qdrant."),
		mcp.WithString("path", mcp.Required(), mcp.Description("Absolute path to the project root")),
	)
	s.AddTool(indexTool, handleIndexProject)

	// 4. Define the Search Tool
	searchTool := mcp.NewTool("codebase_search",
		mcp.WithDescription("Semantic search across the codebase using AI embeddings."),
		mcp.WithString("query", mcp.Description("Natural language query")),
		mcp.WithNumber("limit", mcp.Description("Max results to return")),
	)

	// 5. Add Tool Handler
	s.AddTool(searchTool, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		// A. Extract 'query' using the new type-safe helper (returns string, error)
		query, err := request.RequireString("query")
		if err != nil {
			// MCP Go v0.43+ provides NewToolResultError for graceful argument failures
			return mcp.NewToolResultError(fmt.Sprintf("Invalid or missing 'query': %v", err)), nil
		}

		// B. Extract 'limit' using the type-safe helper with a default value of 20
		// JSON numbers are parsed as float64, so we get it as a float and cast it
		limitFloat := request.GetFloat("limit", 20.0)
		limit := uint64(limitFloat)

		// C. Get embedding from local Ollama
		queryVector, err := getOllamaEmbedding(query)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Ollama error: %v", err)), nil
		}

		// D. Search Qdrant
		searchResults, err := qClient.Query(ctx, &qdrant.QueryPoints{
			CollectionName: CollectionName,
			Query:          qdrant.NewQuery(queryVector...),
			Limit:          &limit,
			WithPayload:    qdrant.NewWithPayload(true),
		})
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Qdrant search error: %v", err)), nil
		}

		// E. Format results for Claude
		var responseText string
		for _, point := range searchResults {
			payload := point.GetPayload()
			filePath := payload["file_path"].GetStringValue()
			lineNum := payload["line_number"].GetIntegerValue()
			codeSnippet := payload["code_snippet"].GetStringValue()

			responseText += fmt.Sprintf("File: %s (Line: %d)\nScore: %.3f\n```go\n%s\n```\n\n---\n",
				filePath, lineNum, point.GetScore(), codeSnippet)
		}

		if responseText == "" {
			responseText = "No relevant code found."
		}

		return mcp.NewToolResultText(responseText), nil
	})

	// Start listening on stdio for VSCode MCP
	fmt.Println("Starting Qdrant MCP Indexer on stdio...")
	if err := server.ServeStdio(s); err != nil {
		log.Printf("serve stdio: %v", err)
	}
}

// Helper: Ensure Collection Exists
func ensureCollection(ctx context.Context, client *qdrant.Client) {
	exists, err := client.CollectionExists(ctx, CollectionName)
	if err != nil {
		log.Fatalf("Error checking collection: %v", err)
	}

	if !exists {
		err = client.CreateCollection(ctx, &qdrant.CreateCollection{
			CollectionName: CollectionName,
			VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
				Size:     VectorDimension,
				Distance: qdrant.Distance_Cosine,
			}),
		})
		if err != nil {
			log.Fatalf("Failed to create collection: %v", err)
		}
		log.Println("Created new Qdrant collection:", CollectionName)
	}
}

// The request payload for Ollama
type OllamaEmbedRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

// Your provided response format (renamed for clarity)
type OllamaEmbedResponse struct {
	Model           string      `json:"model"`
	Embeddings      [][]float64 `json:"embeddings"`
	TotalDuration   int         `json:"total_duration"`
	LoadDuration    int         `json:"load_duration"`
	PromptEvalCount int         `json:"prompt_eval_count"`
}

// getOllamaEmbedding calls the local Ollama API and converts the result for Qdrant
func getOllamaEmbedding(text string) ([]float32, error) {
	// 1. Prepare the request
	reqBody := OllamaEmbedRequest{
		Model: embeddingModel,
		Input: text,
	}

	jsonData, err := sonic.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// 2. Make the HTTP POST to Ollama
	resp, err := http.Post(ollamaURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to call Ollama API: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama API returned status: %d", resp.StatusCode)
	}

	// 3. Decode the response
	var embedResp OllamaEmbedResponse
	if err := sonic.ConfigDefault.NewDecoder(resp.Body).Decode(&embedResp); err != nil {
		return nil, fmt.Errorf("failed to decode ollama response: %w", err)
	}

	// Ensure we actually got an embedding back
	if len(embedResp.Embeddings) == 0 {
		return nil, fmt.Errorf("ollama returned an empty embeddings array")
	}

	// 4. Extract and Convert from []float64 to []float32 for Qdrant
	float64Vector := embedResp.Embeddings[0]
	float32Vector := make([]float32, len(float64Vector))

	for i, val := range float64Vector {
		float32Vector[i] = float32(val)
	}

	return float32Vector, nil
}

// Helper to gracefully fallback if the env var isn't present
func getEnvOrDefault(key, fallback string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return fallback
}

// ---------------------------------------------------------
// Tool Handler: Index Project (Tree-sitter Integration)
// ---------------------------------------------------------

func handleIndexProject(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	rootPath, err := request.RequireString("path")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("Path required: %v", err)), nil
	}

	stats := struct {
		Files   int
		Chunks  int
		Skipped int
		Failed  int
	}{}

	// Initialize Tree-sitter Parser for Go
	parser := sitter.NewParser()
	parser.SetLanguage(golang.GetLanguage())

	// S-Expression Query to find functions and methods
	// Capture the function body (@func) and the name (@name)
	queryStr := `
		(function_declaration
			name: (identifier) @name
		) @func
		(method_declaration
			name: (field_identifier) @name
		) @func
	`
	q, _ := sitter.NewQuery([]byte(queryStr), golang.GetLanguage())

	err = filepath.WalkDir(rootPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Filter directories and non-Go files
		if d.IsDir() {
			if strings.HasPrefix(d.Name(), ".") || d.Name() == "node_modules" || d.Name() == "vendor" {
				return filepath.SkipDir
			}
			return nil
		}
		if filepath.Ext(path) != ".go" {
			stats.Skipped++
			return nil
		}

		// Read File
		content, err := os.ReadFile(path)
		if err != nil {
			stats.Failed++
			return nil
		}
		stats.Files++

		// Parse with Tree-sitter
		tree := parser.Parse(nil, content)
		qc := sitter.NewQueryCursor()
		qc.Exec(q, tree.RootNode())

		// Iterate over matches (Functions/Methods)
		for {
			match, ok := qc.NextMatch()
			if !ok {
				break
			}

			// Extract details
			var funcName, funcBody string
			var startLine, endLine uint32

			for _, capture := range match.Captures {
				node := capture.Node
				name := q.CaptureNameForId(capture.Index)

				if name == "func" {
					funcBody = node.Content(content)
					startLine = node.StartPoint().Row + 1 // 1-based line number for editors
					endLine = node.EndPoint().Row + 1
				} else if name == "name" {
					funcName = node.Content(content)
				}
			}

			// Generate Embedding
			embedding, err := getOllamaEmbedding(funcBody)
			if err != nil {
				stats.Failed++
				continue
			}

			// Create Deterministic UUID (Namespace + FilePath + FunctionName)
			// This ensures re-indexing updates the existing record instead of creating duplicates.
			uniqueID := uuid.NewSHA1(uuid.NameSpaceURL, []byte(path+":"+funcName)).String()

			// Upsert to Qdrant
			pointID := qdrant.NewIDUUID(uniqueID)
			_, err = qClient.Upsert(ctx, &qdrant.UpsertPoints{
				CollectionName: CollectionName,
				Points: []*qdrant.PointStruct{
					{
						Id:      pointID,
						Vectors: qdrant.NewVectors(embedding...),
						Payload: map[string]*qdrant.Value{
							"file_path":    qdrant.NewValueString(path),
							"function":     qdrant.NewValueString(funcName),
							"line_start":   qdrant.NewValueInt(int64(startLine)),
							"line_end":     qdrant.NewValueInt(int64(endLine)),
							"code_snippet": qdrant.NewValueString(funcBody),
						},
					},
				},
			})

			if err != nil {
				log.Printf("Qdrant upsert error: %v", err)
				stats.Failed++
			} else {
				stats.Chunks++
			}
		}

		return nil
	})

	return mcp.NewToolResultText(fmt.Sprintf(
		"Indexing Complete.\nFiles Scanned: %d\nFunctions Indexed: %d\nFailed/Skipped: %d/%d",
		stats.Files, stats.Chunks, stats.Failed, stats.Skipped,
	)), nil
}
