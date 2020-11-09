# KERIS_TRASH
KERIS trash recognition challenge
KERIS 쓰레기 인식 대회

Dataset(데이터셋): `recycle_challenge`
* num_test_images: 3000
* num_test_classes: 8

Pushing dataset (at the root of the folder you want to upload):
데이터셋 푸싱(업로드같은거라 생각하면 됨)(당신이 업로드 하기를 원하는 폴더의 루트에서):

How to run(실행방법):

```bash
nsml run -d recycle_challenge -g 1 --memory 12G --shm-size 2G --cpus 8 -e main.py -a \
         "--mode train --num_epochs 5 --step_size 2"
```

How to list checkpoints saved(리스트 체크포인트 저장 방법):

```bash
nsml model ls keris_admin/recycle_challenge/1
```

How to submit(채점 방법):

```bash
nsml submit keris_admin/recycle_challenge/1 5
```
