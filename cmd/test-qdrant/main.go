package main

import (
	"context"
	"log"
	"time"

	"github.com/qdrant/go-client/qdrant"
)

func main() {
	// 1. Setup a context with a 3-second timeout so the test doesn't hang forever
	// if the server isn't reachable.
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	// 2. Initialize the client targeting the default local gRPC port
	client, err := qdrant.NewClient(&qdrant.Config{
		Host: "localhost",
		Port: 6334,
	})
	if err != nil {
		log.Fatalf("Failed to initialize Qdrant client: %v", err)
	}
	defer client.Close()

	// 3. Make an actual network request to force the gRPC connection
	collections, err := client.ListCollections(ctx)
	if err != nil {
		log.Fatalf("Failed to connect to Qdrant (is the Docker container running?): %v", err)
	}

	// 4. Success!
	log.Printf("✅ Successfully connected to Qdrant! Found %d collections.", len(collections))
}
