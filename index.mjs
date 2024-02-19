import { Textract } from "@aws-sdk/client-textract";
import { DynamoDBClient, PutItemCommand } from "@aws-sdk/client-dynamodb";
import {
  TranslateClient,
  TranslateTextCommand,
} from "@aws-sdk/client-translate";
import { AppSyncClient } from "@aws-sdk/client-appsync";
import OpenAI from "openai";
import graphql, { print } from "graphql";
import gql from "graphql-tag";
import fetch from "node-fetch";
import { performance } from "perf_hooks";

import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { JsonOutputFunctionsParser } from "langchain/output_parsers";

const s3Client = new S3Client({ region: "us-west-1" });

const textractClient = new Textract();
const dynamoDbClient = new DynamoDBClient({ region: "us-west-1" });
const translateClient = new TranslateClient({ region: "us-west-1" });
// const appsyncClient   = new AppSyncClient({ region: "us-west-1" });

const langChainSummary = async (text, language) => {
  const model = new ChatOpenAI({
    openAIApiKey: process.env["OPENAI_API_KEY"],
    temperature: 0.2,
    modelName: "gpt-3.5-turbo",
    topP: 1,
    frequencyPenalty: 0.4,
    presencePenalty: 0.4,
  });

  ////////// Summary //////////
  const prompt1 = PromptTemplate.fromTemplate(
    `For the following text, extract the following information:
    title: A short title that is concise, descriptive, and directly reflective of the summary's main content.
    summary: Summarize the letter, focusing on the main purpose, any offers or requests made. Keep the summary concise, within two sentences.
    action: Any actions that the recipient needs to take in one sentance.
  
    text: {text}`
  );
  const extractionFunctionSchema = {
    name: "extractor",
    description: "For the following text, extract the following information:",
    parameters: {
      type: "object",
      properties: {
        title: {
          type: "string",
          description:
            "A short title that is concise, descriptive, and directly reflective of the summary's main content.",
        },
        summary: {
          type: "string",
          description:
            "Summarize the letter, focusing on the main purpose, any offers or requests made. Keep the summary concise, within two sentences.",
        },
        action: {
          type: "string",
          description:
            "Any actions that the recipient needs to take in one sentance.",
        },
      },
      required: ["title", "summary", "action"],
    },
  };
  const parser = new JsonOutputFunctionsParser({ argsOnly: true });

  const summaryChain = prompt1
    .pipe(
      model.bind({
        functions: [extractionFunctionSchema],
        function_call: { name: "extractor" },
        verbose: true,
        callbacks: [
          {
            handleLLMEnd(output) {
              tokenUsedGPT3 += output.llmOutput.tokenUsage.totalTokens;
            },
          },
        ],
      })
    )
    .pipe(parser);
  const summaryResult = await summaryChain.invoke({ text: text });

  ////////////// Translate //////////////////
  const prompt2 = PromptTemplate.fromTemplate(
    `Translate the title, summary, and action from English to {language}, ensuring that all names and company names remain untranslated.
        title: {title}
        summary: {summary}
        action: {action}
        `
  );
  const translateFunctionSchema = {
    name: "translate",
    description: "Translate the title, summary, and action:",
    parameters: {
      type: "object",
      properties: {
        titleTranslated: {
          type: "string",
          description: "Translate title",
        },
        summaryTranslated: {
          type: "string",
          description: "Translate summary",
        },
        actionTranslated: {
          type: "string",
          description: "Translate action",
        },
      },
      required: ["titleTranslated", "summaryTranslated", "actionTranslated"],
    },
  };

  const translateChain = prompt2
    .pipe(
      model.bind({
        functions: [translateFunctionSchema],
        function_call: { name: "translate" },
        verbose: true,
        callbacks: [
          {
            handleLLMEnd(output) {
              tokenUsedGPT3 += output.llmOutput.tokenUsage.totalTokens;
            },
          },
        ],
      })
    )
    .pipe(parser);

  const translateResult = await translateChain.invoke({
    title: summaryResult.title,
    summary: summaryResult.summary,
    action: summaryResult.action,
    language: language,
  });
  console.log("langchain results:", {
    title: summaryResult.title,
    summary: summaryResult.summary,
    action: summaryResult.action,
    titleTranslated: translateResult.titleTranslated,
    summaryTranslated: translateResult.summaryTranslated,
    actionTranslated: translateResult.actionTranslated,
  });
  return {
    title: summaryResult.title,
    summary: summaryResult.summary,
    action: summaryResult.action,
    titleTranslated: translateResult.titleTranslated,
    summaryTranslated: translateResult.summaryTranslated,
    actionTranslated: translateResult.actionTranslated,
  };
};

const openai = new OpenAI({
  apikey: process.env["OPENAI_API_KEY"],
});
let tokenUsedGPT3 = 0;
let tokenUsedGPT4 = 0;

/////// Extra text and preprocess text //////////////
const filterNonTextualElements = (text) => {
  const urlRegex = /https?:\/\/[^\s]+/g;
  const emailRegex = /\S+@\S+\.\S+/g;
  return text.replace(urlRegex, "").replace(emailRegex, "");
};
const normalizeText = (text) => {
  return text
    .toLowerCase() // or .toUpperCase() based on preference
    .replace(/\s+/g, " ") // Replace multiple spaces with a single space
    .trim(); // Trim leading and trailing spaces
};
const extractTextFromImage = async (bucketName, objectKey) => {
  const params = {
    Document: { S3Object: { Bucket: bucketName, Name: objectKey } },
  };
  const { Blocks } = await textractClient.detectDocumentText(params);
  let extractedText = Blocks.filter((block) => {
    return block.BlockType === "LINE";
  })
    .map((line) => line.Text)
    .join(" ");
  console.log("extractedText", extractedText);
  extractedText = filterNonTextualElements(extractedText);
  extractedText = normalizeText(extractedText);
  return extractedText;
};

const getCompletion = async (
  prompt,
  model = "gpt-3.5-turbo",
  systemPrompt = ""
) => {
  const messages = [
    {
      role: "user",
      content: prompt, // todo imporve prompt, who + type
    },
  ];
  if (systemPrompt) {
    messages.unshift({ role: "system", content: systemPrompt });
  }
  console.log(
    "///////////////////////////////////////////start getCompleteion messages:///////////////////////////////////////////// \n",
    messages
  );
  const response = await openai.chat.completions.create({
    model: model,
    messages: messages,
  });
  if (model === "gpt-3.5-turbo") {
    tokenUsedGPT3 += response["usage"]["total_tokens"];
    console.log(
      "///////////////////////////////////////////GPT 3 token used on this completion:///////////////////////////////////////////// \n",
      response["usage"]["total_tokens"]
    );
  } else {
    tokenUsedGPT4 += response["usage"]["total_tokens"];
    console.log(
      "///////////////////////////////////////////GPT 4 token used on this completion:///////////////////////////////////////////// \n",
      response["usage"]["total_tokens"]
    );
  }

  console.log(
    "getCompleteion response",
    response.choices[0].message?.content.trim()
  );
  return response.choices[0].message?.content.trim();
};

const getImageUrl = async (bucketName, objectKey) => {
  const command = new GetObjectCommand({
    Bucket: bucketName,
    Key: objectKey,
  });

  const signedUrl = await getSignedUrl(s3Client, command, {
    expiresIn: 60 * 5,
  });
  return signedUrl;
};

const gpt4turboImage = async (bucketName, objectKey, language) => {
  try {
    const imageUrl = await getImageUrl(bucketName, objectKey);
    const summaryPromptString =
      `You are a Professional Summarizer.\n` +
      `Perform the following actions: \n` +
      `Please summarize the key points of this letter, focusing on the main purpose, any offers or requests made. Keep the summary concise, within two sentences. \n` +
      `Any actions that the recipient needs to take in one sentance. \n` +
      `A short title that is concise, descriptive, and directly reflective of the summary's main content. \n` +
      `Output in JSON format that contains the following keys: title, summary, action \n`;

    console.log("imageUrl: ", imageUrl);
    const response = await openai.chat.completions.create({
      model: "gpt-4-vision-preview",
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: summaryPromptString,
            },
            {
              type: "image_url",
              image_url: {
                url: imageUrl,
              },
            },
          ],
        },
      ],
      max_tokens: 300,
      temperature: 0.2,
      top_p: 1,
      frequency_penalty: 0.4,
      presence_penalty: 0.4,
    });
    console.log(
      "generateSummaryWithImageGPT4 response: ",
      JSON.stringify(response)
    );
    tokenUsedGPT4 = response["usage"]["total_tokens"];
    const responseContent = response.choices[0].message?.content.trim();
    const { title, summary, action } = JSON.parse(
      responseContent.replace(/```json|```/g, "").trim()
    );

    const translatePrompt =
      `You are a professional translator \n` +
      `Perform the following actions: \n` +
      `Translate the title, summary and action from English to ${language}, ensuring that all names and company names remain untranslated. \n` +
      `Output in JSON format that contains the following keys: titleTranslated, summaryTranslated, actionTranslated \n` +
      `Title: ${title} \n` +
      `Summary: ${summary} \n` +
      `Action: ${action}`;

    const translatedResponseContent = await getCompletion(
      translatePrompt,
      "gpt-4-turbo-preview"
    );
    const { titleTranslated, summaryTranslated, actionTranslated } = JSON.parse(
      translatedResponseContent.replace(/```json|```/g, "").trim()
    );

    return {
      title,
      summary,
      action,
      titleTranslated,
      summaryTranslated,
      actionTranslated,
    };
  } catch (error) {
    console.log("gpt4 failed", error);
    const originalText = await extractTextFromImage(bucketName, objectKey);
    return await generateSummaries(originalText, language);
  }
};

export const handler = async (event) => {
  const bucketName = event.Records[0].s3.bucket.name;
  const objectKey = decodeURIComponent(
    event.Records[0].s3.object.key.replace(/\+/g, " ")
  );
  const userId = objectKey.split("/")[0];
  const documentId = objectKey.split("/").pop().split(".")[0];

  console.log(`bucketName: ${bucketName}`);
  console.log(`objectKey: ${objectKey}`);
  console.log("userId", userId);
  const translatedLanguage = "Chinese";

  // const gpt4turboImageStartTime = performance.now();
  // const {
  //   title,
  //   summary,
  //   action,
  //   titleTranslated,
  //   summaryTranslated,
  //   actionTranslated,
  // } = await gpt4turboImage(bucketName, objectKey, translatedLanguage);
  // const gpt4turboImageEndTime = performance.now();
  // const gpt4turboImageTime = (
  //   gpt4turboImageEndTime - gpt4turboImageStartTime
  // ).toFixed(2);
  // console.log(`gpt4turboImage Execution time: ${gpt4turboImageTime / 1000} s`);
  // const generateSummariesTime = gpt4turboImageTime;

  // ////////////////////////////// test GPT3.5 /////////////////////////////////////////
  // const generateSummariesStartTime = performance.now();
  // await generateSummaries(bucketName, objectKey, translatedLanguage);
  // const generateSummariesEndTime = performance.now();
  // const generateSummariesTotalTime = (
  //   generateSummariesEndTime - generateSummariesStartTime
  // ).toFixed(2);
  // console.log(
  //   `gpt3.5 Execution time: ${generateSummariesTotalTime / 1000} s  \n`,
  //   `gpt3.5 token used: ${tokenUsedGPT3}`
  // );

  // Extract text from image
  const extractTextFromImageStartTime = performance.now();
  const originalText = await extractTextFromImage(bucketName, objectKey);
  const extractTextFromImageEndTime = performance.now();
  const extractTextFromImageTime = (
    extractTextFromImageEndTime - extractTextFromImageStartTime
  ).toFixed(2);
  console.log(
    `extractTextFromImage Execution time: ${extractTextFromImageTime / 1000} s`
  );
  // //   Generate summaries

  const langchainStartTime = performance.now();
  const {
    title,
    summary,
    action,
    titleTranslated,
    summaryTranslated,
    actionTranslated,
  } = await langChainSummary(originalText, translatedLanguage);
  const langchainEndTime = performance.now();
  const langchainTime = (langchainEndTime - langchainStartTime).toFixed(2);
  console.log(`langchain Execution time: ${langchainTime / 1000} s`);

  // create user in database until authantiaciton is implemnted
  await dynamoDbClient.send(
    new PutItemCommand({
      TableName: "UserDocumentSummaries",
      Item: {
        userId: { S: userId },
        documentId: { S: "USER" },
      },
    })
  );
  //
  const now = new Date().toISOString();
  await dynamoDbClient.send(
    new PutItemCommand({
      TableName: "UserDocumentSummaries",
      Item: {
        userId: { S: userId },
        documentId: { S: documentId },
        originalText: { S: originalText },
        title: { S: title },
        summary: { S: summary },
        action: { S: action },
        titleTranslated: { S: titleTranslated },
        summaryTranslated: { S: summaryTranslated },
        actionTranslated: { S: actionTranslated },
        translatedLanguage: { S: translatedLanguage },
        createdAt: { S: now },
        tokenUsedGPT3: { N: tokenUsedGPT3.toString() },
        tokenUsedGPT4: { N: tokenUsedGPT4.toString() },
        generateSummariesTime: { N: langchainTime.toString() },
      },
    })
  );

  tokenUsedGPT3 = 0;
  tokenUsedGPT4 = 0;
  console.log("Successfully processed image and saved summaries to DynamoDB.");
  // for appsync

  const GRAPHQL_ENDPOINT =
    "https://j7lleiiq3rcyncfcxmc4mnxh6a.appsync-api.us-west-1.amazonaws.com/graphql";
  const API_KEY = process.env["APPSYNC_API_KEY"];

  const executeGraphqlMutation = async (query, variables) => {
    const response = await fetch(GRAPHQL_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
      },
      body: JSON.stringify({
        query,
        variables,
      }),
    });

    const responseBody = await response.json();
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`, responseBody);
    }
    return responseBody;
  };

  const documentMutation = `
    mutation NewDocument(
      $documentId: ID!
      $originalText: String!
      $title: String
      $summary: String
      $action: String
      $titleTranslated: String
      $summaryTranslated: String
      $actionTranslated: String
    ) {
      newDocument(
        documentId: $documentId
        originalText: $originalText
        title: $title
        action: $action
        summary: $summary
        titleTranslated: $titleTranslated
        summaryTranslated: $summaryTranslated
        actionTranslated: $actionTranslated
      ) {
        documentId
        originalText
        title
        summary
        action
        titleTranslated
        summaryTranslated
        actionTranslated
      }
    }
  `;
  const variables = {
    documentId: documentId,
    originalText: originalText,
    title: title,
    summary: summary,
    action: action,
    titleTranslated: titleTranslated,
    summaryTranslated: summaryTranslated,
    actionTranslated: actionTranslated,
  };
  try {
    const response = await executeGraphqlMutation(documentMutation, variables);
    console.log("requested mutation response:", response);
    return response;
  } catch (err) {
    console.log("Error requested mutation:", err);
    throw err;
  }
};
