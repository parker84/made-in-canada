# made-in-canada
Repo for building a chat based shopping experience for canadian made products.

## Scraping Products
```sh
# scrape for testing and dump urls
python scrape_products.py \
  --base https://www.roots.com \
  --out ./roots_out \
  --url-regex='\.html$' \
  --limit 500 \
  --dump-urls

# scraping with image download
python scrape_products.py \
  --base https://www.roots.com \
  --out ./roots_out \
  --url-regex='\.html$' \
  --limit 500 \
  --download-images
```