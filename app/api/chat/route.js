import { NextResponse } from "next/server";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Pinecone } from "@pinecone-database/pinecone";

// Define the system prompt with specific instructions for the assistant
const systemPrompt = `
A brief introduction addressing the student's specific request.
Top 3 Professor Recommendations:
    - Professor Name (Subject) - Star Rating
    - Brief summary of the professor's teaching style, strengths, and any relevant details from reviews.
A concise conclusion with any additional advice or suggestions for the student.
For example, when providing recommendations or conclusions, format them as plain text or JSON as required by the user, without unnecessary escape characters.

## Guidelines:
- Always maintain a neutral and objective tone.
- If the query is too vague or broad, ask for clarification to provide more accurate recommendations.
- If no professors match the specific criteria, suggest the closest alternatives and explain why.
- Be prepared to answer follow-up questions about specific professors or compare multiple professors.
- Do not invent or fabricate information. If you don't have sufficient data, state this clearly.
- Respect privacy by not sharing any personal information about professors beyond what's in the official reviews.

Remember, your goal is to help students make informed decisions about their course selections based on professor reviews and ratings.
`;

export async function POST(req) {
  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
  const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
  const index = pc.index('rag').namespace('ns1');

  const data = await req.json();

  // Extract text from the last message
  const text = data[data.length - 1].content;

  // Generate embeddings using the Gemini API
  const embeddingModel = genAI.getGenerativeModel({ model: "text-embedding-004" });
  const embeddingResult = await embeddingModel.embedContent(text);
  const embedding = embeddingResult.embedding;

  // Query Pinecone index with the generated embeddings
  const results = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embedding.values,
  });

  let resultString = 'Returned results:';
  results.matches.forEach((match) => {
    resultString += `\n\nProfessor: ${match.id}\nReview: ${match.metadata.stars}\nSubject: ${match.metadata.subject}\nStars: ${match.metadata.stars}\n`;
  });

  // Prepare the message content for the completion request
  const lastMessage = data[data.length - 1];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

  // Generate the completion using the Gemini model
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
              // Try to parse the content as JSON to prevent double-encoding
              content = JSON.parse(content);
              content = JSON.stringify(content, null, 2); // Format nicely
            } catch (e) {
              // If parsing fails, it means it's already text
              // No need to handle further
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