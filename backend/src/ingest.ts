/* eslint-disable no-process-env */
import { DocumentInterface } from "@langchain/core/documents";
import { RecursiveUrlLoader } from "langchain/document_loaders/web/recursive_url";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Embeddings } from "@langchain/core/embeddings";
import { PostgresRecordManager } from "@langchain/community/indexes/postgres";
import { SitemapLoader } from "langchain/document_loaders/web/sitemap";
import { index } from "./_index.js";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { Chroma } from "@langchain/community/vectorstores/chroma";

/**
 * Load all of the LangSmith documentation via the
 * `RecursiveUrlLoader` and return the documents.
 * @returns {Promise<Array<DocumentInterface>>}
 */
async function loadLangSmithDocs(): Promise<Array<DocumentInterface>> {
  const loader = new RecursiveUrlLoader("https://docs.smith.langchain.com/", {
    maxDepth: 8,
    timeout: 600,
  });
  return loader.load();
}

/**
 * Load all of the LangChain.js API references via
 * the `RecursiveUrlLoader` and return the documents.
 * @returns {Promise<Array<DocumentInterface>>}
 */
async function loadAPIDocs(): Promise<Array<DocumentInterface>> {
  const loader = new RecursiveUrlLoader(
    "https://api.js.langchain.com/index.html/",
    {
      maxDepth: 8,
      timeout: 600,
    }
  );
  return loader.load();
}

/**
 * Load all of the LangChain docs via the sitemap.
 * @returns {Promise<Array<DocumentInterface>>}
 */
async function loadLangChainDocs(): Promise<Array<DocumentInterface>> {
  const loader = new SitemapLoader("https://js.langchain.com/");
  return loader.load();
}

function getEmbeddingsModel(): Embeddings {
  return new OllamaEmbeddings({
    model: "nomic-embed-text",
  });
}

async function ingestDocs() {
  if (
    !process.env.DATABASE_HOST ||
    !process.env.DATABASE_PORT ||
    !process.env.DATABASE_USERNAME ||
    !process.env.DATABASE_PASSWORD ||
    !process.env.DATABASE_NAME
  ) {
    throw new Error(
      "Database constants must be set in the environment"
    );
  }

  const smithDocs = await loadLangSmithDocs();
  console.debug(`Loaded ${smithDocs.length} docs from LangSmith`);
  const apiDocs = await loadAPIDocs();
  console.debug(`Loaded ${apiDocs.length} docs from API`);
  const langchainDocs = await loadLangChainDocs();
  console.debug(`Loaded ${langchainDocs.length} docs from documentation`);

  if (!smithDocs.length) {
    process.exit(1);
  }
  if (!apiDocs.length) {
    process.exit(1);
  }
  if (!langchainDocs.length) {
    process.exit(1);
  }

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkOverlap: 200,
    chunkSize: 4000,
  });
  const docsTransformed = await textSplitter.splitDocuments([
    ...smithDocs,
    ...apiDocs,
    ...langchainDocs,
  ]);

  // We try to return 'source' and 'title' metadata when querying vector store and
  // Weaviate will error at query time if one of the attributes is missing from a
  // retrieved document.

  for (const doc of docsTransformed) {
    if (!doc.metadata.source) {
      doc.metadata.source = "";
    }
    if (!doc.metadata.title) {
      doc.metadata.title = "";
    }
  }

  const embeddings = getEmbeddingsModel();
  const vectorStore = new Chroma(embeddings, {
    collectionName: process.env.COLLECTION_NAME
  });

  const connectionOptions = process.env.RECORD_MANAGER_DB_URL
    ? {
        connectionString: process.env.RECORD_MANAGER_DB_URL,
      }
    : {
        host: process.env.DATABASE_HOST,
        port: Number(process.env.DATABASE_PORT),
        user: process.env.DATABASE_USERNAME,
        password: process.env.DATABASE_PASSWORD,
        database: process.env.DATABASE_NAME,
      };

  const recordManager = new PostgresRecordManager(
    `local/${process.env.COLLECTION_NAME}`,
    {
      postgresConnectionOptions: connectionOptions,
    }
  );
  await recordManager.createSchema();

  const indexingStats = await index({
    docsSource: docsTransformed,
    recordManager,
    vectorStore,
    cleanup: "full",
    sourceIdKey: "source",
    forceUpdate: process.env.FORCE_UPDATE === "true",
  });

  console.log(
    {
      indexingStats,
    },
    "Indexing stats"
  );
}

ingestDocs().catch((e) => {
  console.error("Failed to ingest docs");
  console.error(e);
  process.exit(1);
});
