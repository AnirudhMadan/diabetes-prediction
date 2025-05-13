FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the Streamlit and Flask ports
EXPOSE 8501 5000

# Run both Flask and Streamlit simultaneously using a shell command
CMD ["sh", "-c", "python app.py & python -m streamlit run streamlit_app.py"]
