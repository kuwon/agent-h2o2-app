-- 스키마 보장 (이미 있으면 무시됨)
CREATE SCHEMA IF NOT EXISTS ai;

-- =========================================
-- 1) Customers (ai.kis_customers)
-- =========================================
CREATE TABLE IF NOT EXISTS ai.kis_customers (
  customer_id              VARCHAR(9)  PRIMARY KEY,                  -- 고객번호
  customer_name            TEXT,                                      -- 고객명
  brth_dt                  VARCHAR(8),                                -- 생년월일(YYYYMMDD)
  tot_asst_amt             BIGINT,                                    -- 총자산금액
  cust_ivst_icln_grad_cd   VARCHAR(32),                               -- 고객투자성향등급
  created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),        -- 생성시각
  updated_at               TIMESTAMPTZ                                 -- 수정시각
);

COMMENT ON TABLE  ai.kis_customers IS '고객통합기본';
COMMENT ON COLUMN ai.kis_customers.customer_id            IS '고객번호: Customer ID';
COMMENT ON COLUMN ai.kis_customers.customer_name          IS '고객명: Customer Name';
COMMENT ON COLUMN ai.kis_customers.brth_dt                IS '생년월일(YYYYMMDD): Birth Date';
COMMENT ON COLUMN ai.kis_customers.tot_asst_amt           IS '총자산금액: Total Asset Amount';
COMMENT ON COLUMN ai.kis_customers.cust_ivst_icln_grad_cd IS '고객투자성향등급: Investment Risk Grade';
COMMENT ON COLUMN ai.kis_customers.created_at             IS '생성시각';
COMMENT ON COLUMN ai.kis_customers.updated_at             IS '수정시각';

-- customer_id는 PK이므로 별도 인덱스 불필요


-- =========================================
-- 2) Accounts (ai.kis_accounts)
--   - 복합 PK: (account_id, prd_type_cd)
-- =========================================
CREATE TABLE IF NOT EXISTS ai.kis_accounts (
  account_id            VARCHAR(8)  NOT NULL,                         -- 계좌번호
  customer_id           VARCHAR(9)  NOT NULL,                         -- 고객번호
  acnt_type             TEXT,                                         -- 계좌유형
  prd_type_cd           VARCHAR(4)  NOT NULL,                         -- 상품코드(문자)
  acnt_bgn_dt           VARCHAR(8),                                   -- 계좌개설일자(YYYYMMDD)
  expd_dt               VARCHAR(8),                                   -- 만기일자(YYYYMMDD)
  etco_dt               VARCHAR(8),                                   -- 입사일자(YYYYMMDD)
  rtmt_dt               VARCHAR(8),                                   -- 퇴직일자(YYYYMMDD)
  midl_excc_dt          VARCHAR(8),                                   -- 중간정산일자(YYYYMMDD)
  acnt_evlu_amt         BIGINT,                                       -- 계좌평가액
  copt_year_pymt_amt    BIGINT,                                       -- 회사부담금_연간납입액
  other_txtn_ecls_amt   BIGINT,                                       -- 기타과세제외금액
  rtmt_incm_amt         BIGINT,                                       -- 퇴직소득금액
  icdd_amt              BIGINT,                                       -- 이자/배당금액
  user_almt_amt         BIGINT,                                       -- 사용자부담금
  sbsr_almt_amt         BIGINT,                                       -- 사용자추가납입금
  utlz_erng_amt         BIGINT,                                       -- 운용손익금액
  dfr_rtmt_taxa         BIGINT,                                       -- 이연퇴직소득세
  created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),           -- 생성시각
  updated_at            TIMESTAMPTZ,                                  -- 수정시각
  CONSTRAINT kis_accounts_pkey PRIMARY KEY (account_id, prd_type_cd)
  -- 필요하면 FK 추가 가능:
  -- , CONSTRAINT kis_accounts_customer_fk
  --     FOREIGN KEY (customer_id) REFERENCES ai.kis_customers(customer_id)
);

COMMENT ON TABLE  ai.kis_accounts IS '계좌정보';
COMMENT ON COLUMN ai.kis_accounts.account_id          IS '계좌번호: Account ID';
COMMENT ON COLUMN ai.kis_accounts.customer_id         IS '고객번호: Customer ID';
COMMENT ON COLUMN ai.kis_accounts.acnt_type           IS '계좌유형: Account Type';
COMMENT ON COLUMN ai.kis_accounts.prd_type_cd         IS '상품코드(문자): Product Type Code';
COMMENT ON COLUMN ai.kis_accounts.acnt_bgn_dt         IS '계좌개설일자(YYYYMMDD): Account Open Date';
COMMENT ON COLUMN ai.kis_accounts.expd_dt             IS '만기일자(YYYYMMDD): Expiration Date';
COMMENT ON COLUMN ai.kis_accounts.etco_dt             IS '입사일자(YYYYMMDD): Entering Company Date';
COMMENT ON COLUMN ai.kis_accounts.rtmt_dt             IS '퇴직일자(YYYYMMDD): Retirement Date';
COMMENT ON COLUMN ai.kis_accounts.midl_excc_dt        IS '중간정산일자(YYYYMMDD): Mid Settlement Date';
COMMENT ON COLUMN ai.kis_accounts.acnt_evlu_amt       IS '계좌평가액: Account Evaluation Amount';
COMMENT ON COLUMN ai.kis_accounts.copt_year_pymt_amt  IS '회사부담금_연간납입액: Company Annual Payment Amount';
COMMENT ON COLUMN ai.kis_accounts.other_txtn_ecls_amt IS '기타과세제외금액: Other Tax-excluded Amount';
COMMENT ON COLUMN ai.kis_accounts.rtmt_incm_amt       IS '퇴직소득금액: Retirement Income Amount';
COMMENT ON COLUMN ai.kis_accounts.icdd_amt            IS '이자/배당금액: Interest/Dividend Amount';
COMMENT ON COLUMN ai.kis_accounts.user_almt_amt       IS '사용자부담금: User Allotment Amount';
COMMENT ON COLUMN ai.kis_accounts.sbsr_almt_amt       IS '사용자추가납입금: Subscriber Allotment Amount';
COMMENT ON COLUMN ai.kis_accounts.utlz_erng_amt       IS '운용손익금액: Utilization Earning Amount';
COMMENT ON COLUMN ai.kis_accounts.dfr_rtmt_taxa       IS '이연퇴직소득세: Deferred Retirement Tax';
COMMENT ON COLUMN ai.kis_accounts.created_at          IS '생성시각';
COMMENT ON COLUMN ai.kis_accounts.updated_at          IS '수정시각';

-- 조회 최적화(요청된 index=True 반영 대상)
CREATE INDEX IF NOT EXISTS idx_kis_accounts_customer_id ON ai.kis_accounts (customer_id);
-- account_id는 PK(복합 인덱스의 선두 컬럼)이므로 별도 단일 인덱스는 보통 불필요

