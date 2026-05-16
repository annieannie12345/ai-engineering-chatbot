const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const messagesEl = document.querySelector("#messages");
const sendButton = document.querySelector("#send-button");
const clearButton = document.querySelector("#clear-button");

const history = [];
let activeController = null;
let isGenerating = false;

window.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (isGenerating) {
    stopGeneration();
    return;
  }

  const message = input.value.trim();
  if (!message) return;

  appendMessage("user", message);
  history.push({ role: "user", content: message });
  input.value = "";
  setLoading(true);

  const assistantBubble = appendMessage("assistant", "");
  let assistantText = "";
  activeController = new AbortController();

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: activeController.signal,
      body: JSON.stringify({
        message,
        history: history.slice(-8),
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";

      for (const rawEvent of events) {
        const line = rawEvent.split("\n").find((entry) => entry.startsWith("data: "));
        if (!line) continue;

        const eventData = JSON.parse(line.replace("data: ", ""));
        if (eventData.type === "token") {
          assistantText += eventData.content;
          assistantBubble.textContent = assistantText;
          scrollToLatest();
        }
        if (eventData.type === "error") {
          assistantText = eventData.error;
          assistantBubble.textContent = eventData.error;
        }
      }
    }

    if (assistantText) {
      history.push({ role: "assistant", content: assistantText });
    }
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      if (!assistantText) {
        assistantBubble.textContent = "Generation stopped.";
      }
      return;
    }

    const messageText = error instanceof Error ? error.message : "Unexpected chat error";
    assistantBubble.textContent = messageText;
  } finally {
    activeController = null;
    setLoading(false);
    scrollToLatest();
  }
});

clearButton.addEventListener("click", () => {
  history.length = 0;
  messagesEl.innerHTML = "";
  appendMessage("assistant", "Chat cleared. Ask a new question when you are ready.");
});

function appendMessage(role, content) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "You" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  article.append(avatar, bubble);
  messagesEl.appendChild(article);
  scrollToLatest();
  return bubble;
}

function setLoading(isLoading) {
  isGenerating = isLoading;
  input.disabled = isLoading;
  clearButton.disabled = isLoading;
  sendButton.disabled = false;

  if (isLoading) {
    setSendButtonMode("stop");
    return;
  }

  setSendButtonMode("send");
}

function stopGeneration() {
  if (activeController) {
    activeController.abort();
  }
}

function setSendButtonMode(mode) {
  const isStopMode = mode === "stop";
  sendButton.classList.toggle("primary", !isStopMode);
  sendButton.classList.toggle("stop", isStopMode);
  sendButton.setAttribute("aria-label", isStopMode ? "Stop generating" : "Send message");
  sendButton.setAttribute("title", isStopMode ? "Stop" : "Send");
  sendButton.innerHTML = `<i data-lucide="${isStopMode ? "square" : "send-horizontal"}"></i>`;

  if (window.lucide) {
    window.lucide.createIcons();
  }
}

function scrollToLatest() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}
