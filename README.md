# Fruit Detection

Fruit Detection is a Convolutional Neural Network (CNN) application trained on a dataset of images containing fruits, vegetables, and nuts. This project uses Streamlit for the web interface and Poetry for dependency management.

## Table of Contents

- [Database](#database)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Running Tests](#running-tests)
- [Docker](#docker)
- [License](#license)

## Database

Download the database from kaggle
```kaggle datasets download -d moltean/fruits```

## Installation

To install and set up the project, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/brtenorio/fruit_detection.git
    cd fruit_detection
    ```

2. **Install Poetry (if not already installed):**
    ```sh
    pip install poetry==1.8.3 
    ```

3. **Set up the virtual environment and install dependencies:**
    ```sh
    make all
    ```

## Usage

To run the application, use:

```sh
make run

