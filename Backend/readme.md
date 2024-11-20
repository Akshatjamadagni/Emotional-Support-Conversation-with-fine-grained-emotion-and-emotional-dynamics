
# Backend Integration Testing Environment

A simple backend testing  designed specifically for frontend and backend integration testing.

## Overview

This backend folder contains test scripts for verifying frontend-backend integration. The main component is `test.py` which runs integration test scenarios.

## File Structure
```
backend/
├── test.py          # Main testing 
└── README.md       # This file
```

## Setup

1. Install Python requirements:
```bash
pip install -r requirements.txt
```

2. Run the test script:
```bash
python test.py
```

## Test Cases

The `test.py` script includes basic API endpoint tests:
- GET requests validation
- POST request data handling
- Response format verification
- Error handling scenarios

## Usage

Run all tests:
```bash
python test.py
```




## Quick Start

1. Clone this directory
2. Install dependencies
3. Run test.py
4. Check test results in console output

## Configuration

Default test settings:
- Host: localhost
- Port: 5000
- Timeout: 30s



```

This README.md is focused specifically on the backend testing environment. 