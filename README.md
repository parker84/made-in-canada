# made-in-canada
Repo for building a chat based shopping experience for canadian made products.

## Scraping Products
```sh
source .venv/bin/activate

# roots ✅
python scrape_products.py \
  --base https://www.roots.com \
  --out ./roots_out \
  --use-browser \
  --max-categories 1000 \
  --url-regex='\.html' \
  --download-images \
  --use-postgres

# province of canada ✅
python scrape_products.py \
  --base https://provinceofcanada.com \
  --out ./province_of_canada_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# manmade ✅
python scrape_products.py \
  --base https://manmadebrand.com/ \
  --out ./manmade_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# tilley ✅
python scrape_products.py \
  --base https://ca.tilley.com/ \
  --out ./tilley_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# tentree ✅
python scrape_products.py \
  --base https://www.tentree.com/ \
  --out ./tentree_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# kamik ✅
python scrape_products.py \
  --base https://www.kamik.com/ \
  --out ./kamik_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# sheertex ✅
python scrape_products.py \
  --base https://sheertex.com/ \
  --out ./sheertex_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# baffin ✅
python scrape_products.py \
  --base https://www.baffin.com/ \
  --out ./baffin_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# # ecobee ❌ - not working
# python scrape_products.py \
#   --base https://www.ecobee.com/ \
#   --out ./ecobee_out \
#   --use-browser \
#   --store-type shopify \
#   --max-categories 1000 \
#   --download-images \
#   --use-postgres

# bushbalm ✅
python scrape_products.py \
  --base https://bushbalm.com/ \
  --out ./bushbalm_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# soma chocolate ✅
python scrape_products.py \
  --base https://www.somachocolate.com/ \
  --out ./soma_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# stanfields ✅
python scrape_products.py \
  --base https://www.stanfields.com/ \
  --out ./stanfields_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# balzacs ✅
python scrape_products.py \
  --base https://balzacs.com/ \
  --out ./balzacs_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# muttonhead ✅
python scrape_products.py \
  --base https://muttonheadstore.com/ \
  --out ./muttonhead_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# naked and famous ✅
python scrape_products.py \
  --base https://nakedandfamousdenim.com/ \
  --out ./nakedandfamousdenim_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# regimen lab ✅
python scrape_products.py \
  --base https://regimenlab.ca/ \
  --out ./regimenlab_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# lacanadienne ❌ - not working 
# python scrape_products.py \
#   --base https://lacanadienneshoes.com/ \
#   --out ./lacanadienne_out \
#   --use-browser \
#   --store-type shopify \
#   --max-categories 1000 \
#   --download-images \
#   --use-postgres

# craigs cookies ✅
python scrape_products.py \
  --base https://craigscookies.com/ \
  --out ./craigscookies_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# jenny bird ✅
python scrape_products.py \
  --base https://jenny-bird.ca/ \
  --out ./jennybird_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# green beaver ✅
python scrape_products.py \
  --base https://greenbeaver.com/ \
  --out ./greenbeaver_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# manitobah mucklucks ✅
python scrape_products.py \
  --base https://www.manitobah.ca/ \
  --out ./manitobah_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# moose knuckles ✅
python scrape_products.py \
  --base https://www.mooseknucklescanada.com/ \
  --out ./mooseknuckles_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# reo thompson candies ✅
python scrape_products.py \
  --base https://rheothompson.com/ \
  --out ./reothompson_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# davids tea ✅
python scrape_products.py \
  --base https://davidstea.com/ \
  --out ./davids_tea_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# rocky mountain soap company ✅
python scrape_products.py \
  --base https://www.rockymountainsoap.com/ \
  --out ./rocky_mountain_soap_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# kicking horse coffee
python scrape_products.py \
  --base https://kickinghorsecoffee.ca/ \
  --out ./kicking_horse_coffee_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# st viateur
python scrape_products.py \
  --base https://stviateurbagel.com/ \
  --out ./st_viateur_out \
  --use-browser \
  --store-type shopify \
  --max-categories 1000 \
  --download-images \
  --use-postgres

# # skiis and bikes (maybe - reseller)
# python scrape_products.py \
#   --base https://skiisandbiikes.com/ \
#   --out ./skiisandbikes_out \
#   --use-browser \
#   --store-type shopify \
#   --max-categories 1000 \
#   --download-images \
#   --use-postgres

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