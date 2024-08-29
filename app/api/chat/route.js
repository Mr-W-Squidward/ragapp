import { NextResponse } from "next/server";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Pinecone } from "@pinecone-database/pinecone";

const systemPrompt = `
Your goal is to help students make informed decisions about their course selections based on professor reviews and ratings.
`;

export async function POST(req) {
  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
  const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
  const index = pc.index('rag').namespace('ns1');

  const data = await req.json();

  const text = data[data.length - 1].content;

  const embeddingModel = genAI.getGenerativeModel({ model: "text-embedding-004" });
  const embeddingResult = await embeddingModel.embedContent(text);
  const embedding = embeddingResult.embedding;

  const results = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embedding.values,
  });

  let resultString = 'Returned results:';
  results.matches.forEach((match) => {
    resultString += `\n\nProfessor: ${match.id}\nReview: ${match.metadata.stars}\nSubject: ${match.metadata.subject}\nStars: ${match.metadata.stars}\n`;
  });

  const lastMessage = data[data.length - 1]
  const lastMessageContent = lastMessage.content + resultString
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1)

  const completionModel = genAI.getGenerativeModel({
    model: "gemini-1.5-flash",
    generationConfig: { responseMimeType: "application/json" },
  });
  const response = await completionModel.generateContentStream(systemPrompt + lastMessageContent);

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const chunk of response.stream) {
          let content = chunk.text();
          if (content) {
            try {
              content = JSON.parse(content);
            } catch (e) {
              // If parsing fails, it's text
            }
            const text = encoder.encode(content);
            controller.enqueue(text);
          }
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    },
  });

  return new NextResponse(stream);
}