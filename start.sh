#!/bin/bash
cd /app
python -m src/data_preprocessing
python -m models/image_classification
python -m streamlit run streamlit_app/Home.py --server.port=9000
