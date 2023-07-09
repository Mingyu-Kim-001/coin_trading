dryrun=$1
leverage=$2
budget_allocation=$3
cd $HOME/PycharmProjects/coin_trading
source .env
source ./venv/bin/activate
python3 ./run_trading.py --dryrun "$dryrun" --leverage "$leverage"  --budget_allocation "$budget_allocation" --api_key "$BINANCE_API_KEY" --api_secret "$BINANCE_SECRET_API_KEY" --slack_token "$SLACK_TOKEN"