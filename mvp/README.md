single shot:

// To Generate
python3 single_shot.py   --dataset pretty_test.json   --examples-file examples.json   --out out_chatgpt.json   --model-id gpt-4o-mini   --examples-per-type 3   --example-max-chars 1400   --pretty --progress --debug


// To evaluate
python eval/muc4_embed_eval.py --gold eval/pretty_test.json --pred one-field/predictions_single_field.json --show-matches --pretty

mutli shot: