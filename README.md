# ViQAgent

## Data

The benchmarking data was not uploaded due to its size and autorship. But can be accesed on the corresponding repositories of each dataset. The ViQAgent solution was evaluated on four benchmarks, namely:
- ActivityNet-QA ([Yu, Zhou, et al. 2019](https://github.com/MILVLG/activitynet-qa?tab=readme-ov-file#citation))
- NExT-QA ([Xiao, Junbin, et al. 2021](https://github.com/MILVLG/activitynet-qa?tab=readme-ov-file#citation))
- EgoSchema ([Mangalam, Karttikeya, et al. 2023](https://github.com/egoschema/EgoSchema))
- iVQA ([Yang, Antoine, et al. 2022](https://github.com/antoyang/just-ask/tree/main?tab=readme-ov-file#citation))

These datasets were not used for training, therefore only the test/validation splits were used. Both the NExT-QA and EgoSchema datasets are closed-ended, but the ActivityNet-QA and iVQA datasets don't contain answer options.

### Datasets (Question-Answer)

Store the dataset repositories, in a subfolder named `data`.

<details>
<summary>Click for details on how to download...</summary>

#### ActivityNet-QA

```bash
git clone https://github.com/MILVLG/activitynet-qa ActivityNet_QA
```

#### NExT-QA

```bash
git clone https://github.com/doc-doc/NExT-QA NExT_QA
```

#### EgoSchema

```bash
git clone https://github.com/egoschema/EgoSchema EgoSchema
```

#### iVQA

The dataset is available in a drive folder shared by the authors in [the repository](https://github.com/antoyang/just-ask/tree/main?tab=readme-ov-file#:~:text=We%20provide%20the%20iVQA%20dataset%20at%20this%20link).

Store the dataset in a folder named `iVQA`, inside the `data` folder.

</details>

### Videos

Store the videos of each dataset in the corresponding dataset folder (`data/<DATASET>`), in a subfolder named `videos`.

<details>
<summary>Click for details on how to download...</summary>

#### ActivityNet-QA

Must request the videos to the ActivityNet team, through [this form](http://activity-net.org/download.html#:~:text=Please%20fill%20in%20this%20request%20form%20to%20have%20a%207%2Dday%2Daccess%20to%20download%20the%20videos%20from%20the%20drive%20folders).

#### NExT-QA

The videos are available in a drive folder shared by the authors in [the repository](https://github.com/doc-doc/NExT-QA?tab=readme-ov-file#:~:text=Raw%20videos%20for%20train/val/test%20are%20available).

#### EgoSchema

The videos are available in a drive folder shared by the authors in [the repository](https://github.com/egoschema/EgoSchema#:~:text=Directly%20download%20the%20zipped%20file%20from%20the%20EgoSchema%20Google%20Drive).

#### iVQA

The [dataset](#ivqa)'s column `video_id` contains the video's id in the YouTube platform. The videos can be downloaded using the [`yt-dlp`](https://github.com/yt-dlp) library; for ease, the following script can be used:

```python
import yt_dlp as yt

def download_video(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'format': f'best[height<={720}]',
        'outtmpl': './videos/%(id)s.%(ext)s',  
    }

    with yt.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
```

However, as some of the videos are private or have been removed, the videos can be requested to the authors through the form in the same drive folder shared in [the repository](https://github.com/antoyang/just-ask/tree/main?tab=readme-ov-file#:~:text=We%20provide%20the%20iVQA%20dataset%20at%20this%20link).

</details>
