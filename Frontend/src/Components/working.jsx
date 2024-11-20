import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Button,
  Paper,
  LinearProgress,
  Grid,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  Alert,
  TextField
} from '@mui/material';
import {
  Psychology,
  Architecture,
  Memory,
  Message,
  CheckCircle
} from '@mui/icons-material';
import layer1 from "../Images/layer-1.png"
import layer2 from "../Images/layer-2.png"
import layer3 from "../Images/layer-3.png"
const ModelWorkflow = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [inputText, setInputText] = useState(
    "I feel stuck because I hate my job, but I can't afford to quit. What should I do?"
  );

  const simulateProcessing = (callback) => {
    setProcessing(true);
    setTimeout(() => {
      setProcessing(false);
      callback();
    }, 1000);
  };

  const handleNext = () => {
    simulateProcessing(() => setActiveStep((prev) => prev + 1));
  };

  const handleBack = () => {
    setActiveStep((prev) => prev - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
  };

  const EmotionExtractionStep = () => (
    <Box>
      <Box sx={{ display: 'flex', gap: 2, mb: 4 }}>
        <Box sx={{ flex: 1 }}>
          <TextField
            fullWidth
            multiline
            rows={3}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            label="User Input"
            variant="outlined"
          />
        </Box>
        <Box sx={{ width: 200 }}>
          <img
            src={layer1}
            alt="Cause Aware Encoder"
            style={{ width: '100%', height: 'auto' }}
          />
          <Typography variant="caption" align="center" display="block">
            Cause Aware Encoder
          </Typography>
        </Box>
      </Box>

      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Extracted Emotions
              </Typography>
              <List>
                {[
                  { emotion: 'Anxiety', confidence: 85 },
                  { emotion: 'Frustration', confidence: 65 },
                  { emotion: 'Hopelessness', confidence: 45 }
                ].map((item, idx) => (
                  <ListItem key={idx}>
                    <ListItemText
                      primary={item.emotion}
                      secondary={
                        <Box sx={{ mt: 1 }}>
                          <LinearProgress 
                            variant="determinate" 
                            value={item.confidence}
                          />
                          <Typography variant="caption">
                            Confidence: {item.confidence}%
                          </Typography>
                        </Box>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Identified Causes
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                {[
                  { cause: 'Job Dissatisfaction', type: 'Primary' },
                  { cause: 'Financial Constraints', type: 'Primary' },
                  { cause: 'Career Uncertainty', type: 'Secondary' }
                ].map((item, idx) => (
                  <Paper key={idx} sx={{ p: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="subtitle1">
                        {item.cause}
                      </Typography>
                      <Chip
                        label={item.type}
                        color={item.type === 'Primary' ? 'primary' : 'secondary'}
                        size="small"
                      />
                    </Box>
                  </Paper>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );

  const CausalInteractionStep = () => (
    <Box>
      <Box sx={{ display: 'flex', gap: 2, mb: 4 }}>
        <Box sx={{ flex: 1 }}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Causal Graph Analysis
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    {[
                      {
                        node: 'Job Situation',
                        connections: ['Anxiety', 'Financial Stress'],
                        weight: 0.85
                      },
                      {
                        node: 'Financial Constraints',
                        connections: ['Decision Paralysis', 'Stress'],
                        weight: 0.75
                      },
                      {
                        node: 'Career Goals',
                        connections: ['Motivation', 'Future Planning'],
                        weight: 0.65
                      }
                    ].map((item, idx) => (
                      <Paper key={idx} sx={{ p: 2 }}>
                        <Typography variant="subtitle1" gutterBottom>
                          {item.node}
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                          {item.connections.map((conn, i) => (
                            <Chip
                              key={i}
                              label={conn}
                              size="small"
                              variant="outlined"
                            />
                          ))}
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={item.weight * 100}
                        />
                        <Typography variant="caption">
                          Causal Strength: {(item.weight * 100).toFixed(1)}%
                        </Typography>
                      </Paper>
                    ))}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>
        <Box sx={{ width: 200 }}>
          <img
            src={layer2}
            alt="Causal Interaction Module"
            style={{ width: '100%', height: 'auto' }}
          />
          <Typography variant="caption" align="center" display="block">
            Causal Interaction Module
          </Typography>
        </Box>
      </Box>
    </Box>
  );

  const StrategyExecutionStep = () => (
    <Box>
      <Box sx={{ display: 'flex', gap: 2, mb: 4 }}>
        <Box sx={{ flex: 1 }}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Selected Strategies
                  </Typography>
                  <List>
                    {[
                      { 
                        strategy: 'Empathetic Understanding',
                        probability: 0.85,
                        description: 'Acknowledge and validate feelings'
                      },
                      {
                        strategy: 'Solution Exploration',
                        probability: 0.75,
                        description: 'Guide through options and possibilities'
                      },
                      {
                        strategy: 'Resource Suggestion',
                        probability: 0.65,
                        description: 'Provide relevant resources and tools'
                      }
                    ].map((item, idx) => (
                      <ListItem key={idx}>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                              <Typography>{item.strategy}</Typography>
                              <Typography color="primary">
                                {(item.probability * 100).toFixed(0)}%
                              </Typography>
                            </Box>
                          }
                          secondary={item.description}
                        />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Strategy Execution Metrics
                  </Typography>
                  {[
                    { metric: 'Emotional Resonance', value: 85 },
                    { metric: 'Context Relevance', value: 90 },
                    { metric: 'Action Orientation', value: 75 }
                  ].map((item, idx) => (
                    <Box key={idx} sx={{ mb: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography>{item.metric}</Typography>
                        <Typography>{item.value}%</Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={item.value}
                        color="primary"
                      />
                    </Box>
                  ))}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>
        <Box sx={{ width: 200 }}>
          <img
            src={layer3}
            alt="Strategy Executors"
            style={{ width: '100%', height: 'auto' }}
          />
          <Typography variant="caption" align="center" display="block">
            Strategy Executors
          </Typography>
        </Box>
      </Box>
    </Box>
  );

  const ResponseGenerationStep = () => (
    <Box>
      <Box sx={{ display: 'flex', gap: 2, mb: 4 }}>
        <Box sx={{ flex: 1 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Generated Response
              </Typography>
              <Alert severity="success" sx={{ mb: 4 }}>
                "I understand how challenging it feels to be caught between a job you dislike and financial responsibilities. Let's explore your situation together. What specific aspects of your job are causing the most distress? While we discuss this, we can also look at potential ways to improve your current situation or develop a strategic plan for a future transition that aligns with your financial needs."
              </Alert>
              
              <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
                Response Analysis
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Component Breakdown
                    </Typography>
                    <List>
                      {[
                        'Emotional Acknowledgment',
                        'Situation Exploration',
                        'Strategic Planning',
                        'Financial Consideration'
                      ].map((item, idx) => (
                        <ListItem key={idx}>
                          <ListItemIcon>
                            <CheckCircle color="success" />
                          </ListItemIcon>
                          <ListItemText primary={item} />
                        </ListItem>
                      ))}
                    </List>
                  </Paper>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Response Metrics
                    </Typography>
                    <Box sx={{ '& > *': { mb: 2 } }}>
                      {[
                        { metric: 'Empathy Score', value: 92 },
                        { metric: 'Practicality', value: 85 },
                        { metric: 'Clarity', value: 88 }
                      ].map((item, idx) => (
                        <Box key={idx}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography>{item.metric}</Typography>
                            <Typography>{item.value}%</Typography>
                          </Box>
                          <LinearProgress
                            variant="determinate"
                            value={item.value}
                          />
                        </Box>
                      ))}
                    </Box>
                  </Paper>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Box>
        <Box sx={{ width: 200 }}>
          <img
            src="/api/placeholder/200/150"
            alt="Response Generation"
            style={{ width: '100%', height: 'auto' }}
          />
          <Typography variant="caption" align="center" display="block">
            Response Generation
          </Typography>
        </Box>
      </Box>
    </Box>
  );

  const steps = [
    {
      label: 'Emotion Cause Extraction',
      icon: <Psychology />,
      content: <EmotionExtractionStep />
    },
    {
      label: 'Causal Interaction Module',
      icon: <Architecture />,
      content: <CausalInteractionStep />
    },
    {
      label: 'Strategy Execution',
      icon: <Memory />,
      content: <StrategyExecutionStep />
    },
    {
      label: 'Response Generation',
      icon: <Message />,
      content: <ResponseGenerationStep />
    }
  ];

  return (
    <Card sx={{ width: '100%', maxWidth: '1200px', margin: '0 auto' }}>
      <CardContent>
        <Typography variant="h4" sx={{ mb: 4, textAlign: 'center' }}>
          CauESC Model Workflow
        </Typography>
        
        {processing && (
          <LinearProgress sx={{ mb: 4 }} />
        )}
        
        <Stepper activeStep={activeStep} orientation="vertical">
          {steps.map((step, index) => (
            <Step key={index}>
              <StepLabel StepIconComponent={() => step.icon}>
                <Typography variant="h6">{step.label}</Typography>
              </StepLabel>
              <StepContent>
                <Box sx={{ mt: 2 }}>
                  {step.content}
                  <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                    <Button
                      variant="contained"
                      onClick={handleNext}
                      disabled={processing}
                    >
                      {index === steps.length - 1 ? 'Finish' : 'Continue'}
                    </Button>
                    <Button
                      disabled={index === 0 || processing}
                      onClick={handleBack}
                    >
                      Back
                    </Button>
                  </Box>
                </Box>
              </StepContent>
            </Step>
          ))}
        </Stepper>
        
        {activeStep === steps.length && (
          <Paper square sx={{ p: 2, mt: 2 }}>
          








          <Typography>All steps completed</Typography>
            <Button onClick={handleReset} sx={{ mt: 1 }}>
              Reset
            </Button>
          </Paper>
        )}
      </CardContent>
    </Card>
  );
};

// Add theme customization
export default ModelWorkflow