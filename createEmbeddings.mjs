import { promises as fsp } from "fs";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MongoDBAtlasVectorSearch } from "langchain/vectorstores/mongodb_atlas";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MongoClient } from "mongodb";
import "dotenv/config";

// Console log to check the MongoDB URI, OpenAI API Key, and Test Variable


const client = new MongoClient(process.env.MONGODB_ATLAS_URI || "");
const dbName = "docs";
const collectionName = "embeddings";

try {
  // Connect to the MongoDB client
  await client.connect();
  console.log("Connected successfully to MongoDB server");

  const collection = client.db(dbName).collection(collectionName);

  const docs_dir = "_assets/fcc_docs";
  const fileNames = await fsp.readdir(docs_dir);
  console.log(fileNames);

  for (const fileName of fileNames) {
    const document = await fsp.readFile(`${docs_dir}/${fileName}`, "utf8");
    console.log(`Vectorizing ${fileName}`);

    const splitter = RecursiveCharacterTextSplitter.fromLanguage("markdown", {
      chunkSize: 500,
      chunkOverlap: 50,
    });
    const output = await splitter.createDocuments([document]);

    await MongoDBAtlasVectorSearch.fromDocuments(
      output,
      new OpenAIEmbeddings(),
      {
        collection,
        indexName: "default",
        textKey: "text",
        embeddingKey: "embedding",
      }
    );
  }

  console.log("Done: Closing Connection");
} catch (error) {
  console.error("An error occurred:", error);
} finally {
  // Close the MongoDB client
  await client.close();
  console.log("MongoDB connection closed");
}
