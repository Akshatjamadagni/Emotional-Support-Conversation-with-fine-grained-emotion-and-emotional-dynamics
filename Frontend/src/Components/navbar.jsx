import React from "react";
import { AppBar, Toolbar, Typography, Button, Box } from "@mui/material";
import {
  Home as HomeIcon,
  Chat as ChatIcon,
  DataUsage as DataIcon,
  Work as WorkIcon,
} from "@mui/icons-material";
import { Link } from "react-router-dom"; // Import Link from react-router-dom

const Navbar = () => {
  return (
    <AppBar position="static" color="primary">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          My App
        </Typography>
        <Box sx={{ display: "flex" }}>
          <Button
            color="inherit"
            component={Link}
            to="/"
            startIcon={<HomeIcon />}
          >
            Home
          </Button>
          <Button
            color="inherit"
            component={Link}
            to="/chatbot"
            startIcon={<ChatIcon />}
          >
            Chatbot
          </Button>
          <Button
            color="inherit"
            component={Link}
            to="/dataset"
            startIcon={<DataIcon />}
          >
            Dataset
          </Button>
          <Button
            color="inherit"
            component={Link}
            to="/working"
            startIcon={<WorkIcon />}
          >
            Working
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;
