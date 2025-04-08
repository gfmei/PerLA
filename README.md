# PerLA: Perceptive 3D Language Assistant

Welcome to the official repository for **PerLA (Perceptive 3D Language Assistant)**, accepted by CVPR2025.

## News
- The code is released! Now you can train your customized models!
- The paper has been accepted by CVPR 2025 🔥.
- Code stays tuned for updates!

## About PerLA
**PerLA** is a cutting-edge framework designed to integrate **3D perception** with **natural language understanding**, enabling advanced scene comprehension and interaction capabilities. By leveraging innovative algorithms and models, PerLA bridges the gap between 3D spatial data and language processing to provide state-of-the-art performance in tasks such as:
- 3D question answering
- Dense captioning
- Semantic understanding

Visit the [PerLA website](https://gfmei.github.io/PerLA) to explore more details about the project, methodology, and results.


## Contributing
We welcome and encourage contributions to the PerLA project! If you'd like to contribute:
1. Fork this repository.
2. Create a new branch for your changes.
3. Submit a pull request with a detailed description of your modifications.

## TODO
- [x] Provide code for generate dataset with superpoints
- [x] Provide code for training
- [ ] Provide checkpoints for test


## Usage
Our method builds upon a substantial amount of code from LL3DA, and we gratefully acknowledge the original authors for their valuable contributions.

<details>
  <summary><b>Data Preparation</b></summary>

Our repo requires the 3D data from ScanNet, the natural language annotations, and the pre-trained LLM weights.
Our code requires geometric superpoints.

**Step 1. Download and Prepare the ScanNet 3D Data.**


1. Follow the instructions [here](https://github.com/ch3cook-fdu/Vote2Cap-DETR/tree/master/data/scannet) and download the ScanNetV2 dataset. 
2. Change the `SCANNET_DIR` to the scans folder in [`datasets/scannet/batch_load_scannet_data.py`], and run the following commands.
```{bash}
cd datasets/scannet/
python batch_load_scannet_data.py
```

**Step 2. Prepare Language Annotations**

To train the model, you are required to prepare language annotations from `ScanRefer`, `Nr3D`, `ScanQA`, and the ScanNet part of `3D-LLM`.

1. `ScanRefer`. Follow the commands [here](https://github.com/daveredrum/ScanRefer) to download the `ScanRefer` dataset.
2. `Nr3D`. Follow the commands [here](https://referit3d.github.io/#dataset) to download the `Nr3D` dataset, and [pre-process](https://github.com/ch3cook-fdu/Vote2Cap-DETR/blob/master/data/parse_nr3d.py) it.
3. `ScanQA`. Follow the commands [here](https://github.com/ATR-DBI/ScanQA/blob/main/docs/dataset.md) to download the `ScanQA` dataset.
4. `3D-LLM`. The data are located at [here](dataD_LLM). We have also shared our pre-processing scripts [here](dataD_LLM/pre-process-3D-LLM.py).


Finally, organize the files into the following folders:

```
./data/
  ScanRefer/
    ScanRefer_filtered_train.json
    ScanRefer_filtered_train.txt
    ScanRefer_filtered_val.json
    ScanRefer_filtered_val.txt

  Nr3D/
    nr3d_train.json
    nr3d_train.txt
    nr3d_val.json
    nr3d_val.txt

  ScanQA/
    ScanQA_v1.0_test_w_obj.json
    ScanQA_v1.0_test_wo_obj.json
    ScanQA_v1.0_train.json
    ScanQA_v1.0_val.json

  3D_LLM/
    3d_llm_embodied_dialogue_filtered_train.json
    3d_llm_embodied_dialogue_filtered_val.json
    3d_llm_embodied_planning_filtered_train.json
    3d_llm_embodied_planning_filtered_val.json
    3d_llm_scene_description_train.json
    3d_llm_scene_description_val.json
```

**Step 3. \[Optional\] Download Pre-trained LLM weights.** If your server has no trouble auto-downloading weights from huggingface🤗, feel free to skip this step.

Download files from the `opt-1.3b` checkpoint (or any other decoder-only LLM) at [huggingface](https://huggingface.co/facebook/opt-1.3b/tree/main), and store them under the `./facebook/opt-1.3b` directory. Make sure the required files are downloaded:
```
./facebook/opt-1.3b/
  config.json
  merges.txt
  pytorch_model.bin
  special_tokens_map.json
  tokenizer_config.json
  vocab.json
```

</details>


<details>
  <summary><b>Training</b></summary>

  To train the model as a 3D generalist: 

  ```{bash}
  bash scripts/opt-1.3b/train.generalist.sh
  ```

  After the model is trained, you can tune the model on ScanQA for 3D Question Answering:

  ```{bash}
  bash scripts/opt-1.3b/tuning.scanqa.sh
  ```

  And, on ScanRefer / Nr3D for 3D Dense Captioning:

  ```{bash}
  bash scripts/opt-1.3b/tuning.scanrefer.sh
  bash scripts/opt-1.3b/tuning.nr3d.sh
  ```

  You can also tune the model to predict bounding boxes for open vocabulary object detection!

  ```{bash}
  bash scripts/opt-1.3b/tuning.ovdet.sh
  ```

</details>

<details>
  <summary><b>Evaluation</b></summary>

  To evaluate the model as a 3D generalist:

  ```{bash}
  bash scripts/opt-1.3b/eval.generalist.sh
  ```

  On ScanQA for 3D Question Answering:

  ```{bash}
  bash scripts/opt-1.3b/eval.scanqa.sh
  ```

  And, on ScanRefer / Nr3D for 3D Dense Captioning:

  ```{bash}
  bash scripts/opt-1.3b/eval.scanrefer.sh
  bash scripts/opt-1.3b/eval.nr3d.sh
  ```

</details>



Before contributing, please review our [contribution guidelines](https://github.com/gfmei/PerLA/blob/main/CONTRIBUTING.md).


## Citation
If you find our code or paper useful, please cite
```bibtex
@inproceedings{mei2025PerLA,
  title     = {PerLA: Perceptive 3D language assistant},
  author    = {Guofeng Mei, Wei Lin, Luigi Riz, Yujiao Wu, Fabio Poiesi, Yiming Wang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
```

## Acknowledgments
We extend our gratitude to all contributors and supporters of the PerLA project. Your valuable insights and contributions drive innovation and progress in the field of **3D and language-based AI systems**.

## Contact
For questions, issues, or collaboration opportunities:
- Submit a ticket on the [issues page](https://github.com/gfmei/PerLA/issues).
- Visit the [PerLA project website](https://gfmei.github.io/PerLA/).
- Alternatively, reach out via email: [gmei@fbk.eu](mailto:gmei@fbk.eu).

## Quick Links
- [PerLA Website](https://gfmei.github.io/PerLA/)
- [PerLA Code Repository](https://github.com/gfmei/PerLA)
- [PerLA Paper on arXiv](https://arxiv.org/abs/2411.19774)


## Website License

This project is licensed under the **[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/)**.

[![Creative Commons License](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/)

For more information, visit the [Creative Commons License page](http://creativecommons.org/licenses/by-sa/4.0/).
