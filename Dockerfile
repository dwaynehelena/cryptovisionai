# Use official Python runtime (Full version includes build tools)
FROM python:3.9

# Set work directory
WORKDIR /app

# System dependencies are already included in the full image


# Install TA-Lib
COPY ta-lib-0.4.0-src.tar.gz .
COPY config.guess .
COPY config.sub .
RUN tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cp config.guess ta-lib/config.guess && \
    cp config.sub ta-lib/config.sub && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
