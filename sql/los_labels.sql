-- ============================================
-- Prolonged LOS + 30-day Readmission Labels
-- 运行位置: GCP BigQuery
-- 规则:
--   - 30天起点 = 出院时间 (dischtime)
--   - 再入院 = 下一次住院 (next hadm_id)
--   - 院内死亡不计算再入院
-- ============================================

WITH admissions_seq AS (
    SELECT
        subject_id,
        hadm_id,
        admittime,
        dischtime,
        hospital_expire_flag,
        LEAD(hadm_id) OVER (PARTITION BY subject_id ORDER BY admittime) AS next_hadm_id,
        LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime) AS next_admittime
    FROM `physionet-data.mimiciv_3_1_hosp.admissions`
),
icu_stays AS (
    SELECT 
        icu.stay_id,
        icu.subject_id,
        icu.hadm_id,
        icu.intime,
        icu.outtime,
        adm.dischtime,
        adm.hospital_expire_flag,
        adm.next_hadm_id,
        adm.next_admittime,
        -- 计算ICU住院时长（小时）
        DATETIME_DIFF(icu.outtime, icu.intime, HOUR) as los_hours,
        -- 计算ICU住院时长（天）
        DATETIME_DIFF(icu.outtime, icu.intime, DAY) as los_days
    FROM `physionet-data.mimiciv_3_1_icu.icustays` icu
    JOIN admissions_seq adm
      ON icu.subject_id = adm.subject_id
     AND icu.hadm_id = adm.hadm_id
    WHERE DATETIME_DIFF(icu.outtime, icu.intime, HOUR) >= 24  -- 只保留住院>24h的
)

SELECT 
    i.stay_id,
    i.subject_id,
    i.hadm_id,
    i.los_hours,
    i.los_days,
    
    -- Prolonged LOS 标签 (多个阈值)
    CASE WHEN i.los_days >= 3 THEN 1 ELSE 0 END as prolonged_los_3d,
    CASE WHEN i.los_days >= 5 THEN 1 ELSE 0 END as prolonged_los_5d,
    CASE WHEN i.los_days >= 7 THEN 1 ELSE 0 END as prolonged_los_7d,
    
    -- 30天再入院标签
    CASE
        WHEN i.hospital_expire_flag = 1 THEN NULL
        WHEN i.next_admittime IS NULL THEN 0
        WHEN DATETIME_DIFF(i.next_admittime, i.dischtime, DAY) <= 30 THEN 1
        ELSE 0
    END as readmission_30d
    
FROM icu_stays i
ORDER BY i.stay_id
