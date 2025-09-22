# 한투 연금&퇴직 마스터 Agent

연금 계좌 정보와 관련 정책 정보를 활용한 사내 경진대회용 챗봇 서비스

## Agent App

이 저장소는 다음 구성요소로 이루어진 프로덕션급 에이전틱 시스템의 코드입니다.

1) Streamlit UI  
2) FastAPI 서버  
3) PgVector 확장이 활성화된 Postgres 데이터베이스

> 로컬 Docker 개발 환경과 AWS ECS(Fargate) 프로덕션 환경을 모두 지원합니다.

---

## 목차

- [프로젝트 트리 & 폴더 가이드](#프로젝트-트리--폴더-가이드)
- [빠른 시작 (로컬 개발)](#빠른-시작-로컬-개발)
- [Docker로 실행](#docker로-실행)
- [환경 변수 & 시크릿](#환경-변수--시크릿)
- [데이터 모델 개요](#데이터-모델-개요)
- [UI 개요](#ui-개요)
- [AWS 배포 개요(ECS/Fargate)](#aws-배포-개요ecsfargate)
- [문제 해결(Troubleshooting)](#문제-해결troubleshooting)

---

## 프로젝트 트리 & 폴더 가이드

```
agent-h2o2-app/
├─ .github/workflows/           # CI (테스트/빌드) 워크플로우
├─ agents/                      # 개별 Agent 구현 (tool 사용/프롬프트/정책)
├─ api/                         # FastAPI 앱 (라우팅, healthcheck, backend API)
├─ db/
│  ├─ alembic.ini               # Alembic 설정
│  ├─ alembic/                  # 마이그레이션 스크립트들
│  ├─ session.py                # SQLAlchemy 세션/엔진 생성
│  └─ tables/                   # ORM 모델 (Customers, Accounts, DC 계약 등)
├─ resources/                   # 정책/FAQ/법령 등 원천 자료(인덱싱 전)
├─ scripts/                     # 로컬/배포 자동화 스크립트 (ECR/ECS/ALB 등)
├─ teams/                       # Team 정의(에이전트 오케스트레이션, 라우팅 로직)
├─ ui/
│  ├─ pages/                    # Streamlit 멀티페이지(예: 10_Pension_team.py 등)
│  ├─ panes/                    # 세부 UI 섹션(시뮬/정보/타임라인 등) 모듈
│  ├─ components/               # 공통 UI 컴포넌트(Ag-Grid, 차트 등)
│  ├─ chat.py                   # 채팅 뷰/스트리밍 표시/세션 상태 관리
│  └─ utils/                    # UI 유틸(세션/레이아웃/스타일)
├─ utils/                       # 공용 유틸(로깅, 모델선택, 변환 함수)
├─ workflows/                   # (선택) 워크플로우/파이프라인 정의
├─ workspace/                   # agno workspace 구성(도커 컴포즈/서비스 템플릿)
├─ Dockerfile                   # 앱 컨테이너 빌드 파일
├─ pyproject.toml               # 프로젝트 메타/빌드
├─ requirements.txt             # 의존성(uv/venv 사용 시 참고)
└─ README.md                    # 본 문서
```

### 핵심 디렉터리 설명

- **ui/**: Streamlit 프런트엔드. Ag-Grid 기반 테이블, 생애/정책 타임라인, 시뮬레이션 결과 시각화, 챗 UI를 포함합니다.  
- **api/**: FastAPI 백엔드. `/v1/health` 등 헬스체크와 내부 API를 제공합니다.  
- **db/**: SQLAlchemy ORM 모델과 Alembic 마이그레이션을 관리합니다. 스키마 `ai`에 고객/계좌/DC 계약 테이블과 PgVector 인덱스가 위치합니다.  
- **agents/** & **teams/**: Intent 분류, 정책/FAQ RAG, 개인화(고객/계좌/시뮬레이션 컨텍스트) 등을 조합한 에이전트와 팀 오케스트레이션이 정의됩니다.  
- **scripts/**: ECR 푸시, ECS 서비스/태스크 갱신, ALB/TG 헬스체크 경로 업데이트 등 CI/CD 보조 스크립트가 있습니다.  
- **resources/**: 정책/FAQ/법령 원문(MD/Docx 등). `pgvector` 인덱싱 파이프라인의 입력으로 사용됩니다.  

---

## 빠른 시작 (로컬 개발)

1. **uv 설치** (선택: python 환경 관리)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **개발 환경 세팅**
   ```bash
   ./scripts/dev_setup.sh
   source .venv/bin/activate
   ```

3. **필수 키 설정**
   ```bash
   export OPENAI_API_KEY=YOUR_KEY
   ```

---

## Docker로 실행

```bash
# 워크스페이스 컨테이너 기동
ag ws up
# (기본 포트)
# - Streamlit: http://localhost:8501
# - FastAPI:   http://localhost:8000  (Swagger: /docs)
# - Postgres:  localhost:5432
# 중지
ag ws down
```

---

## 환경 변수 & 시크릿

| 이름 | 예시 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | sk-... | LLM 호출을 위한 키 |
| `RUNTIME_ENV` | `dev` / `prd` | 런타임 환경 플래그 |
| `DB_URL` | `postgresql+psycopg2://ai:***@localhost:5432/ai` | SQLAlchemy 연결 문자열(로컬/서버에 맞게) |
| `TEAM_KEY` | `pension_master` | UI/챗에서 팀 세션 키 구성에 사용 |
| `LOG_LEVEL` | `INFO` | 로그 레벨 |

> 프로덕션에서는 **AWS Secrets Manager**를 사용해 민감정보를 주입하고, **taskRoleArn**과 **executionRoleArn**을 올바르게 설정해 주세요.

---

## 데이터 모델 개요

- **CustomersTable**: 고객 기본 정보(식별자, 성명 등)  
- **AccountsTable**: 계좌/상품(복합 PK 예: `account_id + prd_type_cd`)  
- **DefinedContributionContractTable**: DC 계약(입사/제도가입/중간정산일자, 납입원금, 운용손익, 평가적립 합계 등)  
- **Knowledge(Policy/FAQ)**: PgVector 인덱싱된 정책/FAQ/법령 문서. RAG 검색에 사용.  

> Alembic 마이그레이션으로 스키마 변경을 추적/적용합니다.

---

## UI 개요

- **고객/계좌 선택 패널**: Ag-Grid로 필터/정렬/선택 제공  
- **시뮬레이션 패널**: 납입/수령 옵션별 결과, 타임라인 및 비교표시  
- **정책/FAQ RAG 패널**: 근거 문서 인용, 용어 정의/규정 수치 제시(출처 있을 때만)  
- **챗 패널**: Intent → RAG/DB 조회 → 개인화 응답(생각 출력 숨김, 최종 답만 표시)  

---

## AWS 배포 개요(ECS/Fargate)

- **이미지**: `Dockerfile` 기반으로 ECR에 푸시  
- **ECS 서비스/태스크**: 클러스터 `h2o2-cluster`, 서비스 `h2o2-service`, 태스크 패밀리 `h2o2-task`  
- **네트워킹**: ALB → Target Groups (예: TG-8501 for Streamlit `/`, TG-8000 for FastAPI `/v1/health`)  
- **시크릿/권한**: Secrets Manager → 태스크 정의의 `secrets`로 주입, 적절한 `taskRoleArn`/`executionRoleArn` 필요  
- **헬스체크**: FastAPI `/v1/health`, Streamlit 루트 경로 `/` 권장  

> 배포 스크립트는 `scripts/` 하위에 정리되어 있습니다.

---

## 문제 해결(Troubleshooting)

- **이미지 아키텍처 불일치**: `CannotPullContainerError ... descriptor matching platform 'linux/amd64'` → `docker buildx` 플랫폼 명시 또는 ECR 이미지 아키텍처 확인  
- **taskRoleArn 오류**: “A valid taskRoleArn is not being used” → 태스크 정의에 적절한 IAM Role 연결  
- **ECR 404/Not Found**: `...: not found` → 리포지토리/태그 존재 확인 후 스크립트의 태그 소스/대상 동기화  
- **TLS 인증 실패**: 퍼블릭 레지스트리/미러 사용 시 CA 체인 검증 확인(사내 프록시/미러 환경 고려)  
- **ALB 404/헬스체크 실패**: Target Group의 경로/포트가 컨테이너 포트와 일치하는지 확인(예: 8000 `/v1/health`, 8501 `/`).

---

## 참고

- Streamlit UI, FastAPI, Postgres(PgVector)로 구성되며, 기본 포트는 8501/8000/5432입니다.
- Agno Workspaces 문서: 프로젝트 커스터마이징 및 컨테이너 오케스트레이션에 대한 상세 가이드 참고.

---

_마지막 갱신: 2025-09-22 00:31 _