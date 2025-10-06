import os
import argparse


def evaluate_google_robot(method, folder, step_num, window_size=1, seed_num=3, save_video=False, recompute=False, action_ensemble=True, crop=False, parallel_eval=False, EMA=None, save_attention_map=False):
    for seed in range(seed_num):
        command = f'python -m data.simpler.evaluate --model {method} \
            --model_path {folder} \
            --step {step_num} \
            --window_size {window_size} \
            --seeds {seed}'
        if action_ensemble:
            command += ' --action_ensemble'
        if save_video:
            command += ' --save_video'
        if recompute:
            command += ' --recompute'
        if crop:
            command += ' --crop'
        if EMA is not None:
            command += f' --EMA {EMA}'
        if save_attention_map:
            command += ' --save_attention_map'
        if parallel_eval:
            os.system(f'{command} &')
        else:
            os.system(f'{command}')



if __name__ == '__main__':

    # python scripts/single_task_evaluate.py --benchmark google_robot --seed 3 --folder --step_num 100000
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="simpler")
    parser.add_argument("--method", type=str, default="hypervla")
    parser.add_argument("--folder", type=str, default="")
    parser.add_argument('--step_num', type=int, default=100000)
    parser.add_argument('--seed_num', type=int, default=3)
    parser.add_argument("--save_video", action='store_true', help="save evaluation video or not")
    parser.add_argument('--window_size', type=int, default=2)
    parser.add_argument("--recompute", action='store_true', help="recompute or not")
    parser.add_argument("--action_ensemble", action='store_true', help="use action ensemble or not")
    parser.add_argument("--crop", action='store_true', help="whether to crop the resized image or not")
    parser.add_argument("--save_attention_map", action='store_true', help="whether to save attention map")
    parser.add_argument("--parallel_eval", action='store_true', help="whether to evaluate different seeds in parallel")
    parser.add_argument("--EMA", type=float, default=None, help="evaluate with EMA of model parameters during training")
    args = parser.parse_args()

    if args.benchmark == 'simpler':
        evaluate_google_robot(
            args.method, 
            args.folder, 
            args.step_num, 
            window_size=args.window_size, 
            seed_num=args.seed_num, 
            save_video=args.save_video, 
            recompute=args.recompute,
            action_ensemble=args.action_ensemble,
            crop=args.crop,
            parallel_eval=args.parallel_eval,
            EMA=args.EMA,
            save_attention_map=args.save_attention_map,
        )
