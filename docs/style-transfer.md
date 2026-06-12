# 스타일 트랜스퍼 (AdaIN) 통합

에디터에서 게임 에셋 이미지에 AdaIN 스타일 트랜스퍼를 적용하는 기능.
별도 저장소인 AdaIN 프로젝트(PyTorch)를 로컬 HTTP 서비스로 감싸고,
에디터가 Vite 프록시를 통해 호출한다.

## 구조

```
[에디터 브라우저]                [Vite dev 서버]            [Python 서비스]
🎨 스타일 변환 모달   ─fetch→   /api/style 프록시   ─→   style-service/server.py (:8765)
                                                            ├ adain_service.py  ← ADAIN/net.py, function.py import
                                                            └ config.json       ← 경로·포트 설정
```

- `src/editor/createStyleTransferModal.ts` — 헤더 🎨 버튼 + 모달 UI.
  콘텐츠는 세 가지 — ① 파일 업로드 ② 게임 에셋(`src/assets`의 PNG 목록에서 선택)
  ③ 맵 오브젝트(엔티티 트리에서 나무·분수·건물 등 클릭). 스타일 강도(alpha 0.0~1.0)
  슬라이더, 경계 침식(0~3px), 결과 미리보기, PNG 저장, 그리고 ②③은 **🕹 게임에 적용**과
  **↩ 원본으로 되돌리기**를 지원한다.

## 투명 배경(알파) 보존

콘텐츠에 알파 채널이 있으면(스프라이트 누끼) RGB만 모델에 넣고 — 투명부는 중성 회색으로
합성해 검정 배경 오염을 막는다 — 결과를 **원본 해상도로 되돌린 뒤 원본 알파를 재합성**한
RGBA를 반환한다. 투명 영역이 원본과 비트 단위로 동일하게 유지된다.

- **경계 침식(alpha_erode 0~3px)**: 투명 경계에 스타일 색이 번져 보일 때 알파를 살짝 깎는
  옵션(기본 0). `ImageFilter.MinFilter`(형태학적 침식)라 안티앨리어싱 없이 픽셀아트 경계가
  깔끔하게 유지된다 — cv2 의존성 없음.
- 알파가 없는 이미지(JPG 등)는 기존 경로(RGB, 모델 출력 크기) 그대로다.
- 부수 효과: 알파 있는 에셋은 결과가 원본 크기이므로, 게임 에셋 모드 적용 시 크기가
  달라질 일이 없다.

## 원본으로 되돌리기

첫 적용 때 서비스가 원본을 `style-service/originals/`에 한 번만 시드한다. 이후 몇 번을
다시 적용해도 **↩ 원본으로 되돌리기 한 번에 최초 원본으로 복귀**한다. 버튼은 대상 에셋이
실제 스타일 적용 상태일 때만 활성화된다(`GET /asset-status`로 원본과 현재 파일을 비교).

## 음소거 (에디터 헤더 🔊/🔇)

소리는 게임(iframe)이 낸다 — 토글하면 postMessage(`editor:set-mute`)로 게임의 BGM·효과음을
즉시 끄고, 게임의 오디오 설정(localStorage `my-sample-rpg:audio-settings`의 `isMuted`)에
기록해 게임 리로드·에디터 재시작 후에도 유지된다. 볼륨 슬라이더 값과는 독립(곱 연산)이라
음소거를 해제하면 저장된 볼륨이 그대로 돌아온다.
- `style-service/adain_service.py` — AdaIN 래퍼. **ADAIN 원본 코드는 수정하지 않는다** —
  부작용 없는 `net.py`/`function.py`만 import하고 추론 흐름을 재현한다.
  (`test.py`는 `__main__` 가드 없이 최상위에서 argparse를 실행해 import 불가.)
  모델은 최초 1회만 로드하고, `style_transfer(content_path, style_path, alpha) -> output_path`
  형태의 경로 기반 API도 제공한다.
- `style-service/server.py` — FastAPI 서버. `POST /style-transfer`(전체 이미지),
  `POST /stylize-object`(맵 오브젝트 부분 변환), `POST /apply-asset`(에셋 덮어쓰기),
  `GET /assets`(에셋 목록), `GET /health`. 동기 엔드포인트는 스레드풀에서 실행되어
  추론 중에도 서버가 블로킹되지 않는다.
- `style-service/tile_stylize.py` — 부분 변환 본체: 오브젝트의 타일들을 맵 배치대로 조립 →
  정수배 업스케일 + AdaIN → 타일셋 이미지의 해당 타일 자리에 되써넣기(알파/모양 보존).
- `style-service/asset_store.py` — 쓰기 게이트: `src/assets` 밖 경로 차단, 덮어쓰기 전
  `style-service/backups/`에 타임스탬프 백업.
- `vite.config.ts` — `/api/style` → `http://127.0.0.1:8765` 프록시.

## 오브젝트 자동 누끼 추출 → 스타일 → 게임 반영 파이프라인

게임이 맵을 보고하는 순간(`game:scene-changed`) 에디터가 그 맵의 변환 가능 오브젝트들의
셀 정보를 백그라운드로 서비스에 보내고, 서비스가 타일을 조립해 **투명 배경 PNG로 자동
추출**한다(`style-service/extracted-objects/{id}.png` + 사이드카 `{id}.json`). 타일 원본이
알파를 갖고 있어 별도 배경 제거(세그멘테이션)는 필요 없다.

- 이미 추출된 오브젝트는 스킵(중복 저장 없음). 추출은 비동기라 에디터 UI가 멈추지 않는다.
- 저장 위치가 `src/assets`가 아닌 이유: Vite가 src/ 변경을 감지해 페이지를 리로드하므로,
  맵 인식 때마다 저장하면 "추출 → 리로드 → 재인식" 루프가 된다.
- 모달의 **추출 오브젝트** 탭에서 썸네일을 골라 변환하면(알파 보존·침식 그대로 적용),
  `🕹 게임에 적용`이 사이드카의 셀 정보로 결과를 타일셋에 역패치한다 — 백업·최초 원본
  시드·되돌리기·공유 경고가 모두 기존 흐름과 동일하게 동작한다.
- 같은 타일을 공유하는 오브젝트는 함께 바뀐다(선택 시 ⚠ 경고로 칸 수 표시). 개별 인스턴스만
  바꾸는 것은 범위 외(추후 확장).
- 맵(TMX)을 수정하면 기존 추출본이 낡을 수 있다 — `extracted-objects/`에서 해당 파일을
  지우면 다음 인식 때 다시 추출된다.

## 맵 오브젝트 부분 변환

엔티티 트리에서 나무·분수·가로등·건물 같은 항목(🎨 표시)을 클릭하면 모달이 그 오브젝트를
대상으로 열린다. 에디터가 TMX/TSX를 파싱해 오브젝트가 차지하는 셀과 타일셋 타일 id를
찾아 서비스에 보내고, 서비스는 **그 타일들만** 변환해 타일셋 PNG를 패치한다.

- 타일 군집(좌표 항목)과 영역 오브젝트(이름 항목 — 나무 1, 분수 등) 둘 다 지원.
- 영역 오브젝트는 사각형 안의 "오브젝트 계열" 타일을 모은다. 종류가 다른 타일은 그 공유처가
  전부 같은 종류 객체 안일 때만 포함한다 — 미스라벨된 나무 타일(prop/lamp류 주석)은 살리고,
  사각형에 살짝 겹친 이웃 건물 창문 타일(마을 전체 창문과 공유)이 따라 변하는 오염은 막는다.
  지형(ground/grass 등)과 전역 바닥 채움 타일은 항상 제외된다.
- **같은 타일을 공유하는 곳은 함께 바뀐다** (같은 모양 나무 전부, 같은 타일의 가로등 전부).
  영역 밖 공유 칸이 있으면 모달 배너에 ⚠ 경고로 칸 수를 보여준다.
- 알려진 한계: 한 칸에 오브젝트 타일이 두 겹 쌓인 경우(벽 위에 배너, 천막 아래 소품)
  맨 위 타일만 수집된다 — 아래 타일은 원본 스타일로 남는다.
- NPC·몬스터·표지판·포털은 스프라이트(점 객체)라 이 방식의 대상이 아니다.

## 게임에 적용 (src/assets 덮어쓰기)

`🕹 게임에 적용`은 결과를 `src/assets`의 원본 PNG에 덮어쓴다(게임 에셋 모드: 선택한 파일,
맵 오브젝트 모드: 패치된 타일셋). 안전장치:

- 덮어쓰기 전 원본을 `style-service/backups/<타임스탬프(마이크로초)>__<경로>.png`로 백업한다.
  되돌리려면 그 파일을 원래 자리에 복사하면 된다. 백업+쓰기는 락으로 직렬화된다.
- 서비스는 `src/assets` 밖 경로·PNG 외 형식·존재하지 않는 파일을 거부하고,
  로컬이 아닌 Origin의 POST(드라이브-바이 폼 제출)는 403으로 차단한다.
- 적용하면 Vite가 에셋 변경을 감지해 **게임과 에디터 페이지가 모두 새로고침**된다
  (Vite의 full-reload 브로드캐스트). 모달이 닫히지만 적용은 이미 완료된 상태다.

## 실행 방법

터미널 2개가 필요하다.

```sh
# 터미널 1 — 에디터 (프로젝트 루트)
npm run dev

# 터미널 2 — 스타일 서비스 (최초 1회 의존성 설치 후)
cd style-service
pip install -r requirements.txt
python server.py
```

브라우저에서 `http://localhost:5173/editor.html` → 헤더의 **🎨 스타일 변환** 버튼.
서비스가 꺼져 있으면 모달이 실행 안내 메시지를 보여준다.

## 설정 (경로 하드코딩 없음)

`style-service/config.json` 기본값을 환경변수로 덮어쓸 수 있다.

| config.json | 환경변수 | 기본값 | 설명 |
|---|---|---|---|
| `adainDir` | `ADAIN_DIR` | `../../ADAIN` | AdaIN 프로젝트 폴더 (상대경로는 style-service 기준) |
| `vggPath` | `ADAIN_VGG_PATH` | `models/vgg_normalised.pth` | VGG 인코더 가중치 (adainDir 기준) |
| `decoderPath` | `ADAIN_DECODER_PATH` | `models/decoder.pth` | 디코더 가중치 |
| `host` / `port` | `STYLE_SERVICE_HOST` / `STYLE_SERVICE_PORT` | `127.0.0.1` / `8765` | 서비스 주소 (포트 변경 시 vite.config.ts 프록시도 수정) |

개선판 디코더(`best_decoder.pth.tar`)를 쓰려면 `ADAIN_DECODER_PATH`로 지정하면 된다.

## 동작 세부

- **디바이스**: CUDA가 있으면 GPU, 없으면 CPU 자동 폴백 (`torch.cuda.is_available()`).
  가중치 로드에 `map_location=device`를 지정해 CPU 환경에서도 안전하다.
- **alpha**: feature 공간 블렌딩 비율. `feat = adain_feat * alpha + content_feat * (1 - alpha)`.
  0.0이면 원본 재구성, 1.0이면 완전 스타일화.
- **변환 크기**: 짧은 변 기준 리사이즈(종횡비 유지), 0이면 원본 크기.
  작은 타일(64px 등)은 "원본 크기 유지"가 알맞을 수 있다.
- **동시 요청**: CPU 추론은 락으로 직렬화한다(메모리 보호).

## 의존성 메모

- ADAIN의 `requirements.txt`는 `torch==2.5.1`을 고정하지만, **torch 2.9.1(CPU)에서 추론이
  정상 동작함을 확인했다**(2026-06-12, rpgTile000 + Fall.png, 1.9초/장). 시스템에 torch가
  이미 있으면 다운그레이드 없이 그대로 쓰면 된다.
- torch가 없는 환경이라면 가상환경을 만들어 설치한다. **Python 3.13에서는 torch 2.5.1
  휠이 없으므로**(3.9~3.12 전용) 3.13 지원 버전을 설치한다:
  ```sh
  python -m venv .venv
  .venv\Scripts\activate
  pip install "torch>=2.6" "torchvision>=0.21" "Pillow>=10.2,<13" "numpy>=1.26,<3"
  pip install -r style-service/requirements.txt
  ```
  Python 3.10~3.12 환경이라면 ADAIN 고정 버전(`torch==2.5.1 torchvision==0.20.1`)을
  그대로 써도 된다.
- 서비스 자체 의존성은 `style-service/requirements.txt`의 3개뿐
  (fastapi, uvicorn, python-multipart).
