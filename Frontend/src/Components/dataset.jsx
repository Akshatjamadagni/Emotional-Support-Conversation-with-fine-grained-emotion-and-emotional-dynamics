import React from "react";
import {
  Box,
  Container,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Card,
  CardContent,
  Grid,
  Chip,
  Divider,
} from "@mui/material";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const ESConvStats = () => {
  const datasetStats = [
    {
      category: "# Dialogues",
      train: 1040,
      dev: 130,
      test: 130,
    },
    {
      category: "# Utterances",
      train: 30684,
      dev: 3919,
      test: 3762,
    },
    {
      category: "Avg. Length of Dialogue",
      train: 29.5,
      dev: 30.15,
      test: 28.94,
    },
    {
      category: "Avg. Length of Utterance",
      train: 16.5,
      dev: 16.13,
      test: 15.81,
    },
  ];

  const exampleDialog = [
    { speaker: "seeker", content: "Hello" },
    {
      speaker: "supporter",
      content: "Hello, what would you like to talk about?",
      strategy: "Question",
    },
    {
      speaker: "seeker",
      content:
        "I am having a lot of anxiety about quitting my current job. It is too stressful but pays well",
    },
    {
      speaker: "supporter",
      content: "What makes your job stressful for you?",
      strategy: "Question",
    },
    {
      speaker: "seeker",
      content:
        "I have to deal with many people in hard financial situations and it is upsetting",
    },
  ];

  // Data for the Bar Chart
  const chartData = {
    labels: datasetStats.map((stat) => stat.category),
    datasets: [
      {
        label: "Train",
        data: datasetStats.map((stat) => stat.train),
        backgroundColor: "rgba(63, 81, 181, 0.6)", // Blue
      },
      {
        label: "Dev",
        data: datasetStats.map((stat) => stat.dev),
        backgroundColor: "rgba(255, 152, 0, 0.6)", // Orange
      },
      {
        label: "Test",
        data: datasetStats.map((stat) => stat.test),
        backgroundColor: "rgba(76, 175, 80, 0.6)", // Green
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "ESConv Dataset Statistics (Train/Dev/Test)",
      },
    },
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom color="primary">
        ESConv Dataset Statistics
      </Typography>

      {/* Dataset Statistics Table */}
      <Paper elevation={3} sx={{ mb: 4 }}>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow sx={{ bgcolor: "primary.main" }}>
                <TableCell sx={{ color: "white", fontWeight: "bold" }}>
                  Category
                </TableCell>
                <TableCell align="center" sx={{ color: "white", fontWeight: "bold" }}>
                  Train
                </TableCell>
                <TableCell align="center" sx={{ color: "white", fontWeight: "bold" }}>
                  Dev
                </TableCell>
                <TableCell align="center" sx={{ color: "white", fontWeight: "bold" }}>
                  Test
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {datasetStats.map((row, index) => (
                <TableRow
                  key={index}
                  sx={{ "&:nth-of-type(odd)": { bgcolor: "action.hover" } }}
                >
                  <TableCell component="th" scope="row">{row.category}</TableCell>
                  <TableCell align="center">{row.train}</TableCell>
                  <TableCell align="center">{row.dev}</TableCell>
                  <TableCell align="center">{row.test}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* Dataset Statistics Chart */}
      <Typography variant="h5" gutterBottom color="primary">
        Dataset Statistics Chart
      </Typography>
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Bar data={chartData} options={chartOptions} />
      </Paper>

      {/* Example Conversation */}
      <Typography variant="h5" gutterBottom color="primary">
        Example Conversation
      </Typography>
      <Card elevation={3}>
        <CardContent>
          <Grid container spacing={2} direction="column">
            {exampleDialog.map((message, index) => (
              <Grid item key={index} xs={12}>
                <Box
                  sx={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems:
                      message.speaker === "seeker" ? "flex-end" : "flex-start",
                  }}
                >
                  <Box sx={{ mb: 0.5 }}>
                    <Chip
                      label={message.speaker.toUpperCase()}
                      color={message.speaker === "seeker" ? "primary" : "secondary"}
                      size="small"
                      sx={{ mr: 1 }}
                    />
                    {message.strategy && (
                      <Chip
                        label={message.strategy}
                        color="info"
                        size="small"
                        variant="outlined"
                      />
                    )}
                  </Box>
                  <Paper
                    elevation={1}
                    sx={{
                      p: 2,
                      bgcolor:
                        message.speaker === "seeker"
                          ? "primary.light"
                          : "secondary.light",
                      maxWidth: "80%",
                    }}
                  >
                    <Typography variant="body1">{message.content}</Typography>
                  </Paper>
                </Box>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
    </Container>
  );
};

export default ESConvStats;
