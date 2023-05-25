
import os

#ENV = "DEV"
ENV = "PROD"


## server
host = "0.0.0.0"
port = int(os.environ.get("PORT", 5000))


## info
app_name = "SOCORRO App"
about = ""

## fs
#root = os.path.dirname(os.path.dirname(__file__)) + "/"