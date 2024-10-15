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
  IconButton,
  ThemeProvider,
  createTheme,
  CssBaseline,
} from "@mui/material";
import { Send as SendIcon, Image as ImageIcon } from "@mui/icons-material";

const theme = createTheme({
  palette: {
    primary: {
      main: "#1976d2",
    },
    secondary: {
      main: "#dc004e",
    },
  },
});

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (input.trim()) {
      setMessages([...messages, { text: input, sender: "user" }]);
      // Simulate bot response based on the project report
      setTimeout(() => {
        setMessages((msgs) => [
          ...msgs,
          {
            text: "I'm an AI assistant trained on the CauESC model. How can I provide emotional support today?",
            sender: "bot",
          },
        ]);
      }, 1000);
      setInput("");
    }
  };

  const handleImageUpload = () => {
    // Placeholder for image upload functionality
    console.log("Image upload clicked");
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{ height: "100vh", display: "flex", flexDirection: "column", p: 2 }}
      >
        <Typography variant="h4" component="h1" gutterBottom>
          CauESC: Emotional Support Chatbot
        </Typography>
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
          <IconButton
            color="primary"
            onClick={handleImageUpload}
            sx={{ ml: 1 }}
          >
            <ImageIcon />
          </IconButton>
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
    </ThemeProvider>
  );
};

export default Chatbot;
