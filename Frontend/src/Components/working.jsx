import React from "react";
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  Grid,
  Card,
  CardContent,
} from "@mui/material";

const Working = () => {
  const tasks = [
    { name: "Task 1", progress: 70 },
    { name: "Task 2", progress: 30 },
    { name: "Task 3", progress: 90 },
    { name: "Task 4", progress: 50 },
  ];

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Current Progress
      </Typography>
      <Grid container spacing={3}>
        {tasks.map((task, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {task.name}
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={task.progress}
                  sx={{ height: 10, borderRadius: 5 }}
                />
                <Typography variant="body2" sx={{ mt: 1 }}>
                  {`${task.progress}% Complete`}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default Working;
