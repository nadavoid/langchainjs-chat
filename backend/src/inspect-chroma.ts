// Intended usage: npx ts-node --esm src/inspect-chroma.ts
import { ChromaClient } from "chromadb";

const client = new ChromaClient({
  path: "http://localhost:8000",
});


// We can use a basic embedding function that just returns zeros
// This is fine for inspection purposes
const basicEmbedding = {
  generate: async (texts: string[]) => {
      return texts.map(() => new Array(1536).fill(0))
  }
}

async function inspectChroma() {
  try {
    // List all collections
    const collections = await client.listCollections()
    console.log("Collections:", collections)

      // For each collection, get basic information
    for (const collectionName of collections) {
      const coll = await client.getCollection({ name: collectionName, embeddingFunction: basicEmbedding });

      // Get collection metadata
      const metadata = await coll.metadata;
      console.log(`Collection ${collectionName}:`);
      console.log(`  Description: ${metadata?.description || 'N/A'}`);
      console.log(`  Number of items: ${await coll.count()}`);
      coll

      // // Optional: peek at some items
      // const items = await coll.peek({ limit: 5 });
      // if (items.documents.length > 0) {
      // // if (items.length > 0) {
      //   console.log("  Sample items:", items);
      // } else {
      //   console.log("  No items in this collection.");
      // }

      console.log(); // Add a blank line for readability
    }
  } catch (error) {
      console.error("Error:", error)
  }
}

inspectChroma();
