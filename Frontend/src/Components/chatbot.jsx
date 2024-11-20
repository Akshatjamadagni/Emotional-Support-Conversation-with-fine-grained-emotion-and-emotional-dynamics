import React, { useState, useEffect } from 'react';
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  List,
  ListItem,
  Avatar,
  AppBar,
  Toolbar,
  Chip,
  Badge,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Send as SendIcon,
  FlashOn as FlashOnIcon,
  MoreVert as MoreVertIcon,
  Info as InfoIcon,
  WbSunny as SunIcon,
  DarkMode as MoonIcon,
  LightMode as SunriseIcon
} from '@mui/icons-material';
import axios from 'axios';

const TeamStrikersChat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [greeting, setGreeting] = useState("");

  // Get time-based greeting and icon
  const getGreetingInfo = () => {
    const hour = new Date().getHours();
    if (hour >= 5 && hour < 12) {
      return { text: "Good Morning", icon: <SunriseIcon /> };
    } else if (hour >= 12 && hour < 18) {
      return { text: "Good Afternoon", icon: <SunIcon /> };
    } else {
      return { text: "Good Evening", icon: <MoonIcon /> };
    }
  };

  useEffect(() => {
    const { text } = getGreetingInfo();
    setGreeting(text);
    const initialMessages = [
      {
        text: `${text}! Welcome to Team Strikers Support! 🚀`,
        sender: "bot"
      },
      {
        text: "Our team is ready to assist you with any questions or concerns.",
        sender: "bot"
      }
    ];
    setMessages(initialMessages);
  }, []);

 const handleSend = async () => {
  if (input.trim()) {
    // Add user message to the chat
    setMessages([...messages, { text: input, sender: "user" }]);
    const userMessage = input.trim();
    setInput("");

    try {
      // Simulate bot typing
      setIsTyping(true);

      // Call backend
      const response = await axios.post("http://127.0.0.1:5000/chat", {
        user_input: userMessage
      });

      // Clean the response by removing quotes at start and end if they exist
      const cleanedResponse = response.data.response.replace(/^["']|["']$/g, '');

      // Add bot's response to the chat
      setMessages((msgs) => [
        ...msgs,
        { text: cleanedResponse, sender: "bot" }
      ]);
    } catch (error) {
      setMessages((msgs) => [
        ...msgs,
        { text: "Error communicating with the server. Please try again later.", sender: "bot" }
      ]);
    } finally {
      setIsTyping(false);
    }
  }
};
  return (
    <Box sx={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      <AppBar position="static" sx={{ background: (theme) => theme.palette.primary.main, mb: 2 }}>
        <Toolbar>
          <FlashOnIcon sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Team Strikers Support
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {getGreetingInfo().icon}
            <Badge badgeContent={messages.length} color="error">
              <Chip label="Team Online" color="success" size="small" />
            </Badge>
            <Tooltip title="More options">
              <IconButton color="inherit">
                <MoreVertIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
      </AppBar>

      <Paper
        elevation={3}
        sx={{
          flex: 1,
          mx: 2,
          mb: 2,
          overflow: "auto",
          p: 2
        }}
      >
        <List>
          {messages.map((message, index) => (
            <ListItem
              key={index}
              sx={{
                display: 'flex',
                flexDirection: message.sender === "user" ? "row-reverse" : "row",
                mb: 1
              }}
            >
              <Avatar
                sx={{
                  bgcolor: message.sender === "user" ? "primary.main" : "secondary.main",
                  width: 32,
                  height: 32
                }}
              >
                {message.sender === "user" ? "U" : "T"}
              </Avatar>
              <Paper
                elevation={1}
                sx={{
                  maxWidth: "70%",
                  p: 2,
                  mx: 1,
                  bgcolor: message.sender === "user" ? "primary.light" : "secondary.light",
                  borderRadius: 2
                }}
              >
                <Typography variant="body1">
                  {message.text}
                </Typography>
              </Paper>
            </ListItem>
          ))}
          {isTyping && (
            <ListItem>
              <Avatar sx={{ bgcolor: "secondary.main", width: 32, height: 32 }}>T</Avatar>
              <Paper
                elevation={1}
                sx={{
                  maxWidth: "70%",
                  p: 2,
                  ml: 1,
                  bgcolor: "secondary.light",
                  borderRadius: 2
                }}
              >
                <Typography variant="body2" color="text.secondary">
                  Typing...
                </Typography>
              </Paper>
            </ListItem>
          )}
        </List>
      </Paper>

      <Paper
        elevation={3}
        sx={{
          p: 2,
          mx: 2,
          mb: 2
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleSend()}
            size="small"
          />
          <Button
            variant="contained"
            endIcon={<SendIcon />}
            onClick={handleSend}
            disabled={!input.trim()}
          >
            Send
          </Button>
        </Box>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ mt: 1, display: "block" }}
        >
          <InfoIcon sx={{ fontSize: 16, verticalAlign: 'text-bottom', mr: 0.5 }} />
          Type your message and press Enter or click Send
        </Typography>
      </Paper>
    </Box>
  );
};

export default TeamStrikersChat;
