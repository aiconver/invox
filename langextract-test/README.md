// To Generate
python3 extract_with_langextract.py \
  --dataset pretty_test.json \
  --examples-file examples.json \
  --out out.json \
  --examples-per-type 2 \
  --example-max-per-field 3 \
  --extraction-passes 2 \
  --rpm 6 --pretty --progress


// To evaluate
python3 bleurt_eval.py --gold pretty_test.json --pred out.json \
  --model Elron/bleurt-base-128 --threshold 0.3 --pretty
