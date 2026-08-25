import os
import json
from decimal import Decimal
import psycopg
from utils.config import settings


def _json_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    return str(obj)

QUERY = """
WITH params AS (
    -- SELECT '2026-07-21'::date AS data_emissao
    SELECT CURRENT_DATE AS data_emissao
),
periodo AS (
    SELECT
        %(date_ini)s::date AS data_ini,
        %(date_fim)s::date AS data_fim,
        data_emissao
    FROM params
),
datas AS (
    SELECT gs::date AS dt
    FROM periodo p, generate_series(p.data_ini, p.data_fim, interval '1 day') AS gs
),
 
historico AS (
    SELECT
        d.dt AS rsd_data,
        (SELECT COUNT(*) FROM apartamento WHERE apt_tipapa <> 'SALAO') AS totuh,
 
        (SELECT COUNT(*) FROM diaria
           WHERE 
              dia_datmov = d.dt 
              AND dia_tipmov = 'N' 
              ) AS ocupados,
 
        ( (SELECT COUNT(*) FROM entrada
             WHERE ent_sitent <> 'C' AND ent_datent = d.dt)
        + (SELECT COUNT(*) FROM saida
             WHERE sai_sitsai <> 'C' AND sai_datent = d.dt) ) AS entradas,
 
        (SELECT COUNT(*) FROM saida
           WHERE sai_sitsai <> 'C' AND sai_datsai = d.dt) AS saidas,
 
        (SELECT COUNT(*) FROM diaria
           WHERE 
              dia_datmov = d.dt 
              AND dia_tiphos = 'COR' 
              AND dia_tipmov = 'N' 
              ) AS cort,
 
        (SELECT COUNT(*) FROM diaria
           WHERE 
              dia_datmov = d.dt 
              AND dia_tiphos = 'INT' 
              AND dia_tipmov = 'N' 
              ) AS uso_casa,
 
        (SELECT COUNT(*) FROM diaria
           WHERE 
              dia_datmov = d.dt 
              AND dia_tiphos = 'PER' 
              AND dia_tipmov = 'N'
              ) AS permutas,
 
        (SELECT COUNT(DISTINCT oco_numapa) FROM tab_ocor
           WHERE oco_tipoco = 'MANUT'
             AND d.dt >= oco_datoco::date
             AND d.dt <= COALESCE(oco_datfim::date, oco_datoco::date)
             AND (
                   d.dt < COALESCE(oco_datfim::date, oco_datoco::date)
                   OR oco_horfim IS NULL
                   OR oco_horfim > '12:00:00'
                 )
        ) AS indisponiveis,

        (SELECT COALESCE(SUM(dia_numocu), 0) FROM diaria
           WHERE 
              dia_datmov = d.dt
              AND dia_tipmov = 'N'
              ) AS pax,
 
        (SELECT COALESCE(SUM(dia_numcri), 0) FROM diaria
           WHERE 
              dia_datmov = d.dt 
              AND dia_tipmov = 'N'
              ) AS chd,
 
        (SELECT COALESCE(SUM(dia_valdia), 0) FROM diaria
           WHERE 
              dia_datmov = d.dt 
              AND dia_tipmov = 'N'
              ) AS receita,

        (SELECT COALESCE(SUM(dia_valdia), 0) FROM diaria
           WHERE 
              dia_datmov = d.dt 
              AND dia_tipmov = 'A') AS acrescimo,

        (SELECT COALESCE(SUM(sai_vldesdia), 0) FROM saida
           WHERE 
              sai_datsai = d.dt 
              AND sai_sitsai <> 'C') AS desconto,

        (SELECT COUNT( * ) FROM reserva
           WHERE
              res_sitres <> 'C'
              AND res_tipres = 'BL'
              AND res_datent = d.dt) AS bloqueada
 
    FROM datas d, params
    WHERE d.dt < params.data_emissao
),
 
previsao AS (
    SELECT
        d.dt AS rsd_data,
        (SELECT COUNT(*) FROM apartamento WHERE apt_tipapa <> 'SALAO') AS totuh,

        ( (SELECT COUNT(*) FROM entrada
             WHERE ent_sitent <> 'C'
               AND d.dt >= ent_datent AND d.dt < ent_dtpresai)
        + (SELECT COUNT(*) FROM reserva
             WHERE res_sitres <> 'C'
               AND res_datent >= params.data_emissao
               AND d.dt >= res_datent AND d.dt < res_datsai
               AND (res_numapa IS NOT NULL OR res_numapa <> '')) ) AS ocupados,

        (SELECT COUNT(*) FROM reserva
             WHERE res_sitres <> 'C'
               AND res_datent >= params.data_emissao
               AND res_datent = d.dt
               AND (res_numapa IS NOT NULL OR res_numapa <> '')) AS entradas,

        ( (SELECT COUNT(*) FROM entrada
             WHERE ent_sitent <> 'C' AND ent_dtpresai = d.dt)
        + (SELECT COUNT(*) FROM reserva
             WHERE res_sitres <> 'C'
               AND res_datent >= params.data_emissao
               AND res_datsai = d.dt
               AND (res_numapa IS NOT NULL OR res_numapa <> '')) ) AS saidas,

        ( (SELECT COUNT(*) FROM entrada
             WHERE ent_sitent <> 'C' AND ent_tiphos = 'COR'
               AND d.dt >= ent_datent AND d.dt < ent_dtpresai)
        + (SELECT COUNT(*) FROM reserva
             WHERE res_sitres <> 'C' AND res_tiphos = 'COR'
               AND res_datent >= params.data_emissao
               AND d.dt >= res_datent AND d.dt < res_datsai
               AND (res_numapa IS NOT NULL OR res_numapa <> '')) ) AS cort,

        ( (SELECT COUNT(*) FROM entrada
             WHERE ent_sitent <> 'C' AND ent_tiphos = 'INT'
               AND d.dt >= ent_datent AND d.dt < ent_dtpresai)
        + (SELECT COUNT(*) FROM reserva
             WHERE res_sitres <> 'C' AND res_tiphos = 'INT'
               AND res_datent >= params.data_emissao
               AND d.dt >= res_datent AND d.dt < res_datsai
               AND (res_numapa IS NOT NULL OR res_numapa <> '')) ) AS uso_casa,

        ( (SELECT COUNT(*) FROM entrada
             WHERE ent_sitent <> 'C' AND ent_tiphos = 'PER'
               AND d.dt >= ent_datent AND d.dt < ent_dtpresai)
        + (SELECT COUNT(*) FROM reserva
             WHERE res_sitres <> 'C' AND res_tiphos = 'PER'
               AND res_datent >= params.data_emissao
               AND d.dt >= res_datent AND d.dt < res_datsai
               AND (res_numapa IS NOT NULL OR res_numapa <> '')) ) AS permutas,

        (SELECT COUNT(*) FROM apartamento WHERE apt_tipapa <> 'SALAO' AND apt_sitapa = 'MAN') AS indisponiveis,

        ( (SELECT COALESCE(SUM(ent_numocu), 0) FROM entrada
             WHERE ent_sitent <> 'C'
               AND d.dt >= ent_datent AND d.dt < ent_dtpresai)
        + (SELECT COALESCE(SUM(res_numocu), 0) FROM reserva
             WHERE res_sitres <> 'C'
               AND res_datent >= params.data_emissao
               AND d.dt >= res_datent AND d.dt < res_datsai
               AND (res_numapa IS NOT NULL OR res_numapa <> '')) ) AS pax,

        ( (SELECT COALESCE(SUM(ent_numcri), 0) FROM entrada
             WHERE ent_sitent <> 'C'
               AND d.dt >= ent_datent AND d.dt < ent_dtpresai)
        + (SELECT COALESCE(SUM(res_numcri), 0) FROM reserva
             WHERE res_sitres <> 'C'
               AND res_datent >= params.data_emissao
               AND d.dt >= res_datent AND d.dt < res_datsai
               AND (res_numapa IS NOT NULL OR res_numapa <> '')) ) AS chd,

        ( (SELECT COALESCE(SUM(ent_val1dia), 0) FROM entrada
             WHERE ent_sitent <> 'C'
               AND d.dt >= ent_datent AND d.dt < ent_dtpresai)
        + (SELECT COALESCE(SUM(res_val1dia), 0) FROM reserva
             WHERE res_sitres <> 'C'
               AND res_datent >= params.data_emissao
               AND d.dt >= res_datent AND d.dt < res_datsai
               AND (res_numapa IS NOT NULL OR res_numapa <> '')) ) AS receita,

        0 AS acrescimo,

        0 AS desconto,

        (SELECT COUNT( * ) FROM reserva
           WHERE
              res_sitres <> 'C'
              AND res_tipres = 'BL'
              AND res_datent >= params.data_emissao
              AND res_datent = d.dt) AS bloqueada

    FROM datas d, params
    WHERE d.dt >= params.data_emissao
),

por_dia AS (
    SELECT *, 'Historico' AS periodo FROM historico
    UNION ALL
    SELECT *, 'Previsao'  AS periodo FROM previsao
)
 
SELECT
    rsd_data                                                                                         AS "Data",
    ROUND(((totuh - indisponiveis) - ocupados - bloqueada::numeric) / NULLIF(totuh, 0) * 100, 2)     AS "%% / UH Disp",
    ROUND(ocupados::numeric / NULLIF(totuh, 0) * 100, 2)                                             AS "%% / UH Total",
    totuh                                                                                            AS "Total",
    ocupados                                                                                         AS "Ocupados",
    entradas                                                                                         AS "Entrada",
    saidas                                                                                           AS "Saidas",
    cort                                                                                             AS "Cort",
    uso_casa                                                                                         AS "Uso da Casa",
    permutas                                                                                         AS "Permutas",
    indisponiveis                                                                                    AS "Indisponiveis",
    (totuh - indisponiveis - ocupados - bloqueada)                                                   AS "Disponiveis",
    pax                                                                                              AS "# Pax",
    chd                                                                                              AS "# CHD",
    receita                                                                                          AS "Receita de Hospedagem",
    ROUND(receita::numeric / NULLIF(ocupados, 0), 2)                                                 AS "Diaria Media",
    acrescimo                                                                                        AS "Acrescimo",
    ROUND(receita::numeric / NULLIF(totuh, 0), 2)                                                    AS "RevPar",
    periodo                                                                                          AS "Periodo",
    desconto                                                                                         AS "Desconto",
    bloqueada                                                                                        AS "Bloqueada"
FROM por_dia
ORDER BY rsd_data;
"""

OUT = "data/sql_extract.jsonl"


def main():
    os.makedirs("data", exist_ok=True)

    try:
        conn = psycopg.connect(
          host=settings.pg_host,
          port=settings.pg_port,
          user=settings.pg_user,
          password=settings.pg_password,
          dbname=settings.pg_database,
          connect_timeout=5,
        )
    except psycopg.OperationalError as e:
        print(f"FALHA AO CONECTAR no banco ({settings.pg_host}):")
        print(f"  {e}")
        raise

    try:
        with conn:
            with conn.cursor(name="extract_cursor") as cur:
                cur.itersize = 5000
                try:
                    cur.execute(QUERY, {"date_ini": "2026-08-01", "date_fim": "2026-08-30"})
                except psycopg.errors.SyntaxError as e:
                    print("FALHA NA QUERY SQL (sintaxe):")
                    print(f"  {e}")
                    raise

                columns = [desc.name for desc in cur.description]
                with open(OUT, "w", encoding="utf-8") as f:
                    count = 0
                    for row in cur:
                        record = dict(zip(columns, row))
                        f.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")
                        count += 1
                        if count % 100 == 0:
                            print(f"  {count} linhas extraídas...")

        print(f"OK: {count} linhas exportadas para {OUT}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
