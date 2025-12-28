# made-in-canada
Repo for building a chat based shopping experience for canadian made products.

## Scraping Products
```sh
source .venv/bin/activate
python scrape_products.py \
  --base https://www.roots.com \
  --out ./roots_out \
  --use-browser \
  --max-categories 1000 \
  --url-regex='\.html' \
  --use-postgres

# scrape products and download images
python scrape_products.py \
  --base https://www.roots.com \
  --out ./roots_out \
  --use-browser \
  --max-categories 1000 \
  --url-regex='\.html' \
  --download-images \
  --use-postgres

# province of canada
python scrape_products.py \
  --base https://provinceofcanada.com \
  --out ./province_of_canada_out \
  --use-browser \
  --store-type shopify \
  --max-categories 300 \
  --download-images \
  --use-postgres
```

## Running the App
```sh
source .venv/bin/activate
export OPENAI_API_KEY=...
streamlit run app.py
```