// server.js
import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";
import axios from "axios";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

app.get("/", (req, res) => res.send("✅ AI BizChain Backend is running"));

// 🧠 Common fallback function
async function generateWithOpenRouter(prompt, type) {
  const response = await axios.post(
    "https://openrouter.ai/api/v1/chat/completions",
    {
      model: "mistralai/mistral-7b-instruct",
      messages: [
        { role: "system", content: "You are a creative marketing AI." },
        { role: "user", content: prompt },
      ],
    },
    {
      headers: {
        "Authorization": `Bearer ${process.env.OPENROUTER_API_KEY}`,
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "AI-BizChain",
        "Content-Type": "application/json",
      },
    }
  );

  return response.data.choices[0].message.content.trim();
}

// 🧠 Generate Content
app.post("/generate", async (req, res) => {
  const { description, type } = req.body;
  if (!description || !type) {
    return res.status(400).json({ error: "Missing description or type" });
  }

  const prompt = `
You are an expert marketing content creator and brand strategist. 
Your task is to craft a highly engaging, original, and audience-focused ${type} based on the following business description:

"${description}"

Guidelines:
- The tone should be persuasive, creative, and aligned with modern digital marketing trends.
- Ensure the message feels personalized and emotionally resonant.
- Highlight the business’s unique value proposition clearly.
- Include a strong call-to-action (if appropriate for the ${type}).
- Keep it concise yet impactful.
`;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        { role: "system", content: "You are a creative AI content generator for marketing and business." },
        { role: "user", content: prompt },
      ],
      max_tokens: 300,
    });

    const content = completion.choices[0].message.content.trim();
    return res.json({ content });
  } catch (err) {
    console.error("⚠️ OpenAI failed, trying OpenRouter:", err.message);

    try {
      const content = await generateWithOpenRouter(prompt, type);
      return res.json({ content });
    } catch (fallbackErr) {
      console.error("❌ Both OpenAI and OpenRouter failed:", fallbackErr.message);
      return res.status(500).json({ error: "AI content generation failed." });
    }
  }
});

// 🧠 Refine Content
app.post("/refine", async (req, res) => {
  const { currentContent, feedback, type } = req.body;

  if (!currentContent || !feedback || !type) {
    return res.status(400).json({ error: "Missing current content, feedback, or type" });
  }

  const prompt = `
You are a skilled marketing copy editor. 
Refine the following ${type} based on the user's feedback.

--- Current Content ---
${currentContent}

--- User Feedback ---
${feedback}

Please provide an improved version that aligns with the feedback while keeping the message clear, engaging, and on-brand.
`;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        { role: "system", content: "You are an expert marketing assistant helping to refine creative content." },
        { role: "user", content: prompt },
      ],
      max_tokens: 300,
    });

    const refined = completion.choices[0].message.content.trim();
    res.json({ refined });
  } catch (err) {
    console.error("⚠️ OpenAI refine failed, trying OpenRouter:", err.message);

    try {
      const refined = await generateWithOpenRouter(prompt, type);
      res.json({ refined });
    } catch (fallbackErr) {
      console.error("❌ Both OpenAI and OpenRouter refine failed:", fallbackErr.message);
      res.status(500).json({ error: "Refinement failed on both APIs." });
    }
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 Server running on http://localhost:${PORT}`));
