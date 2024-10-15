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
  Grid,
  Card,
  CardContent,
  Chip,
  Tooltip,
  CircularProgress,
} from "@mui/material";
import {
  Send as SendIcon,
  Image as ImageIcon,
  Psychology as PsychologyIcon,
  SentimentSatisfiedAlt as EmotionIcon,
  Lightbulb as StrategyIcon,
} from "@mui/icons-material";

const theme = createTheme({
  palette: {
    primary: {
      main: "#1976d2",
    },
    secondary: {
      main: "#dc004e",
    },
    background: {
      default: "#f5f5f5",
    },
  },
});

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);

  const handleSend = () => {
    if (input.trim()) {
      setMessages([...messages, { text: input, sender: "user" }]);
      setInput("");
      setIsTyping(true);
      setShowWelcome(false);

      // Simulate bot response based on the project report
      setTimeout(() => {
        setMessages((msgs) => [
          ...msgs,
          {
            text: "I'm here to provide emotional support using causal awareness techniques. How can I assist you today?",
            sender: "bot",
            cause: "User inquiry",
            strategy: "Empathetic response",
          },
        ]);
        setIsTyping(false);
      }, 2000);
    }
  };

  const handleImageUpload = () => {
    // Placeholder for image upload functionality
    console.log("Image upload clicked");
  };

  const examplePrompts = [
    "I'm feeling overwhelmed with work lately.",
    "I had an argument with my friend and I'm not sure what to do.",
    "I'm excited about a new opportunity but also nervous.",
    "I've been feeling down and I'm not sure why.",
  ];

  const WelcomeScreen = () => (
    <Box sx={{ textAlign: "center", mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        Welcome to CauESC: Emotional Support Chatbot
      </Typography>
      <Typography variant="body1" paragraph>
        I'm here to provide emotional support using advanced causal awareness
        techniques.
      </Typography>
      <Grid container spacing={3} justifyContent="center" sx={{ mt: 2 }}>
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <PsychologyIcon color="primary" sx={{ fontSize: 40 }} />
              <Typography variant="h6">Cause Aware</Typography>
              <Typography variant="body2">
                I can identify the root causes of emotional distress.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <EmotionIcon color="secondary" sx={{ fontSize: 40 }} />
              <Typography variant="h6">Emotionally Intelligent</Typography>
              <Typography variant="body2">
                I use causal interaction to understand your emotions deeply.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <StrategyIcon color="success" sx={{ fontSize: 40 }} />
              <Typography variant="h6">Strategic Support</Typography>
              <Typography variant="body2">
                I employ various strategies to provide effective emotional
                support.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      <Typography variant="h6" sx={{ mt: 4, mb: 2 }}>
        Try asking me about:
      </Typography>
      <Box
        sx={{
          display: "flex",
          flexWrap: "wrap",
          justifyContent: "center",
          gap: 1,
        }}
      >
        {examplePrompts.map((prompt, index) => (
          <Chip
            key={index}
            label={prompt}
            onClick={() => {
              setInput(prompt);
              setShowWelcome(false);
            }}
            sx={{ m: 0.5 }}
          />
        ))}
      </Box>
    </Box>
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{ height: "100vh", display: "flex", flexDirection: "column", p: 2 }}
      >
        <Typography
          variant="h4"
          component="h1"
          gutterBottom
          sx={{ display: "flex", alignItems: "center" }}
        >
          CauESC: Emotional Support Chatbot
        </Typography>
        <Paper elevation={3} sx={{ flex: 1, mb: 2, overflow: "auto", p: 2 }}>
          {showWelcome ? (
            <WelcomeScreen />
          ) : (
            <List>
              {messages.map((message, index) => (
                <ListItem
                  key={index}
                  alignItems="flex-start"
                  sx={{
                    flexDirection:
                      message.sender === "user" ? "row-reverse" : "row",
                    mb: 2,
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
                      p: 2,
                      ml: 1,
                      mr: 1,
                      backgroundColor:
                        message.sender === "user"
                          ? "primary.light"
                          : "secondary.light",
                      borderRadius: "20px",
                    }}
                  >
                    <Typography variant="body1">{message.text}</Typography>
                    {message.sender === "bot" && (
                      <Box sx={{ mt: 1 }}>
                        <Tooltip title="Identified cause of emotion">
                          <Chip
                            icon={<PsychologyIcon />}
                            label={message.cause}
                            size="small"
                            sx={{ mr: 1 }}
                          />
                        </Tooltip>
                        <Tooltip title="Support strategy used">
                          <Chip
                            icon={<StrategyIcon />}
                            label={message.strategy}
                            size="small"
                          />
                        </Tooltip>
                      </Box>
                    )}
                  </Paper>
                </ListItem>
              ))}
            </List>
          )}
          {isTyping && (
            <Box sx={{ display: "flex", justifyContent: "flex-start", mt: 2 }}>
              <CircularProgress size={20} />
              <Typography variant="body2" sx={{ ml: 1 }}>
                CauESC is thinking...
              </Typography>
            </Box>
          )}
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
