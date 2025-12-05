document.addEventListener("DOMContentLoaded", function () {
  const chatContainer = document.getElementById("chatContainer");
  const chatInput = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");

  // Hàm thêm tin nhắn vào giao diện
  function appendMessage(text, sender = "user") {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("chat-message", sender === "bot" ? "bot-message" : "user-message");

    const msg = document.createElement("div");
    msg.classList.add("message", "p-3", "rounded", "shadow-sm");
    msg.classList.add(sender === "bot" ? "bg-light" : "bg-primary", sender === "bot" ? "text-dark" : "text-white");
    msg.innerText = text;

    messageDiv.appendChild(msg);
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  // Gửi tin nhắn
  sendBtn.addEventListener("click", async function () {
    const text = chatInput.value.trim();
    if (!text) return;

    appendMessage(text, "user");
    chatInput.value = "";

    // Gọi API Flask
    try {
      const baseURL = `${window.location.origin}`;
      const res = await fetch(`${baseURL}/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });


      if (!res.ok) throw new Error("Server error!");

      const data = await res.json();
      appendMessage(data.reply || "Không có phản hồi từ server.", "bot");
    } catch (err) {
      appendMessage("⚠️ Lỗi khi kết nối server!", "bot");
      console.error(err);
    }
  });

  // Gửi khi nhấn Enter
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendBtn.click();
    }
  });
});
