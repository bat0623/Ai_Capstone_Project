#!/bin/bash

# (1) 디버그 모드 활성화: 실행되는 명령어를 화면에 모두 표시합니다.
set -x

echo "SCRAPING ..."

(
  # data 디렉터리로 이동
  cd data

  # seriesDir 반복
  for seriesDir in ./*/
  do
    if [[ "$seriesDir" != "./Test/" && "$seriesDir" != "./ALL/" ]]; then
      (
        cd "$seriesDir"
        # gameDir 반복
        for gameDir in ./*/
        do
          (
            cd "$gameDir"
            # (2) echo로 단순히 출력만 하던 부분을, 실제 명령을 실행하도록 변경
            echo "### Now in: $gameDir ###"

            # 원본: echo "mkdir -p raw" → 실제 실행
            mkdir -p raw

            # 원본: echo "python3 scraper.py" → 실제 실행
            python3 scraper.py
          )
        done
      )
    fi
  done
)

echo "PARSING ..."

(
  cd processing

  # 원본: echo "python3 parseRawData.py" → 실제 실행
  python3 parseRawData.py

  # 원본: echo "python3 getStatistics.py" → 실제 실행
  python3 getStatistics.py
)

