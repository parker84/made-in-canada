# made-in-canada
Repo for building a chat based shopping experience for canadian made products.

## Scraping Products
```sh
python scrape_products.py \
  --base https://www.roots.com \
  --out ./roots_out \
  --use-browser \
  --max-categories 10 \
  --url-regex='\.html'

# scrape products and download images
python scrape_products.py \
  --base https://www.roots.com \
  --out ./roots_out \
  --use-browser \
  --max-categories 100 \
  --url-regex='\.html' \
  --download-images
```