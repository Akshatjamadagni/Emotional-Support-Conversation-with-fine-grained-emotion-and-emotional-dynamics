import React, { useState } from "react";
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  List,
  ListItem,

  Avatar,
} from "@mui/material";
import { Send as SendIcon } from "@mui/icons-material";

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (input.trim()) {
      setMessages([...messages, { text: input, sender: "user" }]);
      // Simulate bot response
      setTimeout(() => {
        setMessages((msgs) => [
          ...msgs,
          { text: "I'm a bot response!", sender: "bot" },
        ]);
      }, 1000);
      setInput("");
    }
  };

  return (
    <Box
      sx={{ height: "80vh", display: "flex", flexDirection: "column", p: 2 }}
    >
      <Paper elevation={3} sx={{ flex: 1, mb: 2, overflow: "auto", p: 2 }}>
        <List>
          {messages.map((message, index) => (
            <ListItem
              key={index}
              alignItems="flex-start"
              sx={{
                flexDirection:
                  message.sender === "user" ? "row-reverse" : "row",
              }}
            >
              <Avatar
                sx={{
                  bgcolor:
                    message.sender === "user"
                      ? "primary.main"
                      : "secondary.main",
                }}
              >
                {message.sender === "user" ? "U" : "B"}
              </Avatar>
              <Paper
                elevation={1}
                sx={{
                  maxWidth: "70%",
                  p: 1,
                  ml: 1,
                  mr: 1,
                  backgroundColor:
                    message.sender === "user"
                      ? "primary.light"
                      : "secondary.light",
                }}
              >
                <Typography variant="body1">{message.text}</Typography>
              </Paper>
            </ListItem>
          ))}
        </List>
      </Paper>
      <Box sx={{ display: "flex" }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Type a message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && handleSend()}
        />
        <Button
          variant="contained"
          endIcon={<SendIcon />}
          onClick={handleSend}
          sx={{ ml: 1 }}
        >
          Send
        </Button>
      </Box>
    </Box>
  );
};

export default Chatbot;
