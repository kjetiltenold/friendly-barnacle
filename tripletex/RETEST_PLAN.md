# Retest Plan

Use this after deploying the latest agent changes.

## Priority Order

Retest in this order:

1. Supplier invoice voucher
2. Accounting dimensions + voucher
3. Timesheet + project invoice
4. Create project
5. Travel expense
6. Salary / payroll
7. Create invoice (simple)
8. Order + invoice + payment
9. Create employee
10. Create supplier

Reasoning:

- `supplier invoice` and `accounting dimensions` were the clearest local failures before the last schema fixes, and they now produce materially better voucher payloads.
- `timesheet + project invoice` now completes locally end-to-end within the iteration budget.
- `project` and `travel` still rely more on executor auto-fill, so they should be validated early.
- `salary` is more stable locally now, but still worth confirming in the sandbox.

## Exact Prompts

### 1. Supplier invoice voucher

```text
Wir haben die Rechnung INV-2026-6392 vom Lieferanten Silberberg GmbH (Org.-Nr. 871719500) über 6500 NOK einschließlich MwSt. erhalten. Der Betrag betrifft Bürodienstleistungen (Konto 6860). Erfassen Sie die Lieferantenrechnung mit der korrekten Vorsteuer (25 %).
```

### 2. Accounting dimensions + voucher

```text
Opprett en fri regnskapsdimensjon "Marked" med verdiene "Privat" og "Bedrift". Bokfør deretter et bilag på konto 6300 for 12650 kr, knyttet til dimensjonsverdien "Privat".
```

### 3. Timesheet + project invoice

```text
Registrer 5 timer for Ingrid Nilsen (ingrid.nilsen@example.org) på aktiviteten "Analyse" i prosjektet "Plattformintegrasjon" for Bergvik AS (org.nr 989231898). Timesats: 1400 kr/t. Generer en prosjektfaktura til kunden basert på de registrerte timene.
```

### 4. Create project

```text
Créez le projet "Implémentation Montagne" lié au client Montagne SARL (nº org. 989074784). Le chef de projet est Lucas Robert (lucas.robert@example.org).
```

### 5. Travel expense

```text
Register a travel expense for Charles Harris (charles.harris@example.org) for "Client visit Oslo". The trip lasted 5 days with per diem (daily rate 800 NOK). Expenses: flight ticket 2300 NOK and taxi 500 NOK.
```

### 6. Salary / payroll

```text
Führen Sie die Gehaltsabrechnung für Mia Hoffmann (mia.hoffmann@example.org) für diesen Monat durch. Das Grundgehalt beträgt 40350 NOK. Fügen Sie einen einmaligen Bonus von 7350 NOK zum Grundgehalt hinzu.
```

### 7. Create invoice (simple)

```text
Opprett og send en faktura til kunden Bergvik AS (org.nr 890733751) på 28900 kr eksklusiv MVA. Fakturaen gjelder Systemutvikling.
```

### 8. Order + invoice + payment

```text
Erstellen Sie einen Auftrag für den Kunden Grünfeld GmbH (Org.-Nr. 920238882) mit den Produkten Datenberatung (5628) zu 23000 NOK und Cloud-Speicher (1573) zu 16550 NOK. Wandeln Sie den Auftrag in eine Rechnung um und registrieren Sie die vollständige Zahlung.
```

### 9. Create employee

```text
We have a new employee named Charles Taylor, born 21. October 1994. Please create them as an employee with email charles.taylor@example.org and start date 3. June 2026.
```

### 10. Create supplier

```text
Registre el proveedor Dorada SL con número de organización 853166553. Correo electrónico: faktura@doradasl.no.
```

## What To Record

For each retest, capture:

1. Contest score before and after the rerun.
2. Whether the task completed with zero 4xx errors.
3. The relevant `/solve` request and response logs.
4. Any Tripletex `requestId` values from failed calls.

## Triage Notes

If a retest still fails:

1. Check whether the failure was due to missing nested payload fields.
2. Check whether the model chose the wrong VAT type family, especially outgoing vs incoming VAT.
3. Check whether the executor auto-fill path should be promoted into an explicit dedicated tool requirement.
4. Prefer fixing the executor or tool schema over adding more prompt prose.
