dryrun=$1
leverage=$2
budget_allocation=$3
budget_keep=$4
script_dir=$(dirname "$0")
cd "$script_dir" || exit
source ./env_file
source ./venv/bin/activate
echo `date`
python3 ./run_trading.py --dryrun "$dryrun" --leverage "$leverage"  --budget_allocation "$budget_allocation" --budget_keep "$budget_keep" --api_key "$BINANCE_API_KEY" --api_secret "$BINANCE_SECRET_API_KEY" --slack_token "$SLACK_TOKEN"
