# made-in-canada
Repo for building a chat based shopping experience for canadian made products.

## Scraping Products
```sh
source .venv/bin/activate

# roots ✅
python scrape_products.py \
  --base https://www.roots.com \
  --use-browser \
  --max-categories 1000 \
  --url-regex='\.html' \
  --use-postgres

# province of canada ✅
python scrape_products.py \
  --base https://provinceofcanada.com \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# manmade ✅
python scrape_products.py \
  --base https://manmadebrand.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# tilley ✅
python scrape_products.py \
  --base https://ca.tilley.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# tentree ✅
python scrape_products.py \
  --base https://www.tentree.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# kamik ✅
python scrape_products.py \
  --base https://www.kamik.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 2 \
  --use-postgres

# sheertex ✅
python scrape_products.py \
  --base https://sheertex.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# baffin ✅
python scrape_products.py \
  --base https://www.baffin.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# # ecobee ❌ - not working
# python scrape_products.py \
#   --base https://www.ecobee.com/ \
#   --use-browser \
#   --store-type shopify \
#   --max-categories 1000 \
# #   --use-postgres

# bushbalm ✅
python scrape_products.py \
  --base https://bushbalm.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# soma chocolate ✅
python scrape_products.py \
  --base https://www.somachocolate.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# stanfields ✅
python scrape_products.py \
  --base https://www.stanfields.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# balzacs ✅
python scrape_products.py \
  --base https://balzacs.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# muttonhead ✅
python scrape_products.py \
  --base https://muttonheadstore.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# naked and famous ✅
python scrape_products.py \
  --base https://nakedandfamousdenim.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# regimen lab ✅
python scrape_products.py \
  --base https://regimenlab.ca/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# lacanadienne ❌ - not working 
# python scrape_products.py \
#   --base https://lacanadienneshoes.com/ \
#   --use-browser \
#   --store-type shopify \
#   --max-categories 1000 \
# #   --use-postgres

# craigs cookies ✅
python scrape_products.py \
  --base https://craigscookies.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# jenny bird ✅
python scrape_products.py \
  --base https://jenny-bird.ca/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# green beaver ✅
python scrape_products.py \
  --base https://greenbeaver.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# manitobah mucklucks ✅
python scrape_products.py \
  --base https://www.manitobah.ca/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# moose knuckles ✅
python scrape_products.py \
  --base https://www.mooseknucklescanada.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# reo thompson candies ✅
python scrape_products.py \
  --base https://rheothompson.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# davids tea ✅
python scrape_products.py \
  --base https://davidstea.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# rocky mountain soap company ✅
python scrape_products.py \
  --base https://www.rockymountainsoap.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# kicking horse coffee ✅
python scrape_products.py \
  --base https://kickinghorsecoffee.ca/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# st viateur ✅
python scrape_products.py \
  --base https://stviateurbagel.com/ \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --use-postgres

# # skiis and bikes (maybe - reseller)
# python scrape_products.py \
#   --base https://skiisandbiikes.com/ \
#   --use-browser \
#   --store-type shopify \
#   --max-categories 1000 \
# #   --use-postgres

# Brands
# TODO: Canada Goose
# TODO: Lululemon
# TODO: Mejuri
# TODO: Grohmann Knives
# TODO: Aritzia
# TODO: Kotn
# TODO: Herschel
# TODO: Joe Fresh
# TODO: purdy's chocolatier

# Re-sellers (maybe)
# TODO: Canadian Tire
# TODO: Sport Chek
# TODO: Mountain Equipment Co-op
# TODO: Simons
```

## Running the App
```sh
source .venv/bin/activate
export OPENAI_API_KEY=...
streamlit run app.py
```