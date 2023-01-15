# Contributing

해당 레퍼지토리에 기여할 때 누가 어떤 내용에 기여했는지 구분하기 위해 아래의 규칙을 지키기.

## Commit Process

1. `main` 브랜치는 최종본만 업로드합니다.
2. `dev` 브랜치에 실질적으로 필요한 모든 코드를 업로드합니다. 기능 개발, 실험은 feat-{기능명}, exp-{실험명}와 같은 브랜치를 만들어서 push하고 dev에 merge합니다.
3. 눈갱 방지를 위해 최소한 코드 에디터가 제공하는 [코드 정리(Linting)](https://code.visualstudio.com/docs/python/linting) 기능을 사용하여 *(VSCode 의 경우 cmd+k+f)* [PEP8](https://peps.python.org/pep-0008/)을 준수하려고 노력해 봐요.
4. git commit convention
    - fix: 버그 패치
    - feat: 새로운 기능을 추가할 경우
    - exp: 실험 추가
    - style: 코드의 변화와 관련없는 코드 포맷 변경, 세미 콜론 누락, 코드 수정이 없는 경우
    - refactor: 기능 변경 없이 레거시를 리팩터링하는 거대한 작업
    - docs: 문서와 관련하여 수정한 부분이 있을 때 사용
    - tests: test를 추가하거나 수정했을 때를 의미
    - chore: build와 관련된 부분, 빌드 태스크 업데이트, 패키지 매니저 설정 등 여러가지 production code와 무관한 부분들을 의미
    - first commit


5. __이슈__, __PR__ 역시 제목은 같은 형식으로 작성합니다.
    
```text
<convention>: <한 줄 설명> (이슈링크)
# 한 줄 띄우기
<커밋에 대한 본문(필수 사항) 설명(한글)>

```

`example`
```text
fix: eda index error fix (#102)

- 박스친 이미지를 출력하는 과정에서 생긴 인덱스 문제를 해결
- train.py에서 함수 호출하는 방식 변경
```


### 참고
- [UDACITY Convention ](https://yunyoung1819.tistory.com/m/160)
- [CONTRIBUTING.md template](https://gist.github.com/PurpleBooth/b24679402957c63ec426)
- [VSCode Linting](https://code.visualstudio.com/docs/python/linting)
- [Python PEP8](https://peps.python.org/pep-0008/)
- [Git commit convention](https://www.conventionalcommits.org/ko/v1.0.0/)
- [Git commit convention reference: FastAPI Template project](https://github.com/tiangolo/full-stack-fastapi-postgresql)
- [Sementic versioning](https://semver.org/lang/ko/)
- [Repository convention reference: SSAFY BookFlex project](https://github.com/glenn93516/BookFlex)