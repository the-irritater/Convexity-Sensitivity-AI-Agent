"""
Generate Validation Workbook
============================
Creates a comprehensive, premium Excel workbook with formula-driven cells
verifying bond pricing, duration, convexity, portfolio aggregation,
VaR calculations, and Nelson-Siegel curve fitting.
"""

import os
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from src.utils import (
    load_bond_portfolio, load_yield_curve, load_monte_carlo,
    generate_bond_cashflows, DATA_DIR, OUTPUT_DIR
)
from src.part1_analytics import BondAnalytics
from src.part2_yield_curve import NelsonSiegelSvensson

def create_validation_workbook():
    print("Generating validation_workbook.xlsx...")
    
    # Create workbook
    wb = openpyxl.Workbook()
    
    # Common styles
    font_family = "Segoe UI"
    
    title_font = Font(name=font_family, size=16, bold=True, color="1B365D")
    section_font = Font(name=font_family, size=12, bold=True, color="1B365D")
    header_font = Font(name=font_family, size=10, bold=True, color="FFFFFF")
    bold_font = Font(name=font_family, size=10, bold=True)
    regular_font = Font(name=font_family, size=10)
    italic_font = Font(name=font_family, size=9, italic=True, color="555555")
    
    header_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
    accent_fill = PatternFill(start_color="F0F4F8", end_color="F0F4F8", fill_type="solid")
    zebra_fill = PatternFill(start_color="F7F9FA", end_color="F7F9FA", fill_type="solid")
    green_fill = PatternFill(start_color="E6F4EA", end_color="E6F4EA", fill_type="solid")
    
    thin_border = Border(
        left=Side(style='thin', color='D3D3D3'),
        right=Side(style='thin', color='D3D3D3'),
        top=Side(style='thin', color='D3D3D3'),
        bottom=Side(style='thin', color='D3D3D3')
    )
    
    double_bottom_border = Border(
        top=Side(style='thin', color='1B365D'),
        bottom=Side(style='double', color='1B365D')
    )

    align_center = Alignment(horizontal='center', vertical='center')
    align_left = Alignment(horizontal='left', vertical='center')
    align_right = Alignment(horizontal='right', vertical='center')
    
    # Formatting strings
    fmt_currency = "₹#,##0.00"
    fmt_currency_cr = "₹#,##0.00\" Cr\""
    fmt_percent = "0.00%"
    fmt_number = "#,##0.0000"
    fmt_integer = "#,##0"

    # Load data
    df_bonds = load_bond_portfolio()
    df_yc = load_yield_curve()
    df_mc = load_monte_carlo()

    # ═════════════════════════════════════════════════════════════
    # TAB 1: Bond Pricing Validation
    # ═════════════════════════════════════════════════════════════
    ws1 = wb.active
    ws1.title = "Bond Pricing Validation"
    ws1.views.sheetView[0].showGridLines = True
    
    ws1.cell(row=2, column=2, value="Bond Pricing Validation (First Principles)").font = title_font
    ws1.cell(row=3, column=2, value="Step-by-step cash flow discounting for 5 representative bonds").font = italic_font
    
    # Selected bonds for verification
    selected_bond_ids = ["ZTFI-0001", "ZTFI-0003", "ZTFI-0005", "ZTFI-0012", "ZTFI-0021"]
    
    curr_row = 5
    for bond_id in selected_bond_ids:
        bond_row = df_bonds[df_bonds['BondID'] == bond_id].iloc[0]
        
        # Bond info table
        ws1.cell(row=curr_row, column=2, value=f"Bond Details: {bond_id}").font = section_font
        curr_row += 1
        
        info_labels = ["Face Value", "Coupon Rate", "Coupon Freq", "Maturity (Yrs)", "YTM", "CSV Dirty Price", "Excel Discounted Price"]
        info_values = [
            bond_row['FaceValue'],
            bond_row['CouponRate'],
            bond_row['CouponFrequency'],
            bond_row['YearsToMaturity'],
            bond_row['YieldToMaturity'],
            bond_row['DirtyPrice'],
            None # Will write formula
        ]
        
        # Write info box
        for col_idx, (lbl, val) in enumerate(zip(info_labels, info_values), start=2):
            ws1.cell(row=curr_row, column=col_idx, value=lbl).font = bold_font
            ws1.cell(row=curr_row, column=col_idx).fill = accent_fill
            ws1.cell(row=curr_row, column=col_idx).alignment = align_center
            ws1.cell(row=curr_row, column=col_idx).border = thin_border
            
            val_row = curr_row + 1
            if lbl == "Excel Discounted Price":
                # Formula written later below cashflows
                pass
            else:
                cell = ws1.cell(row=val_row, column=col_idx, value=val)
                cell.font = regular_font
                cell.alignment = align_center
                cell.border = thin_border
                if "Rate" in lbl or "YTM" in lbl:
                    cell.number_format = fmt_percent
                elif "Price" in lbl or "Value" in lbl:
                    cell.number_format = fmt_currency
                elif "Freq" in lbl:
                    cell.number_format = fmt_integer
                else:
                    cell.number_format = fmt_number
                    
        # Cash flow table headers
        curr_row += 3
        headers = ["Period", "Time (t)", "Cash Flow", "Discount Factor", "Present Value"]
        for col_idx, h in enumerate(headers, start=2):
            cell = ws1.cell(row=curr_row, column=col_idx, value=h)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = align_center
            cell.border = thin_border
            
        cf, times = generate_bond_cashflows(
            bond_row['FaceValue'], bond_row['CouponRate'],
            int(bond_row['CouponFrequency']), bond_row['YearsToMaturity']
        )
        
        cf_start_row = curr_row + 1
        ytm_cell_ref = f"${get_column_letter(6)}${val_row}" # YTM column is 6 (F)
        freq_cell_ref = f"${get_column_letter(4)}${val_row}" # Freq column is 4 (D)
        
        for i, (t_val, cf_val) in enumerate(zip(times, cf), start=1):
            curr_row += 1
            # Period
            c_per = ws1.cell(row=curr_row, column=2, value=i)
            c_per.font = regular_font
            c_per.alignment = align_center
            c_per.border = thin_border
            # Time
            c_t = ws1.cell(row=curr_row, column=3, value=t_val)
            c_t.font = regular_font
            c_t.alignment = align_right
            c_t.number_format = fmt_number
            c_t.border = thin_border
            # Cash Flow
            c_cf = ws1.cell(row=curr_row, column=4, value=cf_val)
            c_cf.font = regular_font
            c_cf.alignment = align_right
            c_cf.number_format = fmt_currency
            c_cf.border = thin_border
            # Discount Factor Formula: 1 / (1 + YTM / Freq) ^ (t * Freq)
            c_df = ws1.cell(row=curr_row, column=5, value=f"=1 / (1 + {ytm_cell_ref} / {freq_cell_ref}) ^ (C{curr_row} * {freq_cell_ref})")
            c_df.font = regular_font
            c_df.alignment = align_right
            c_df.number_format = fmt_number
            c_df.border = thin_border
            # Present Value Formula: Cash Flow * Discount Factor
            c_pv = ws1.cell(row=curr_row, column=6, value=f"=D{curr_row} * E{curr_row}")
            c_pv.font = regular_font
            c_pv.alignment = align_right
            c_pv.number_format = fmt_currency
            c_pv.border = thin_border
            
        cf_end_row = curr_row
        
        # Total Row
        curr_row += 1
        ws1.cell(row=curr_row, column=2, value="Total").font = bold_font
        ws1.cell(row=curr_row, column=2).alignment = align_left
        ws1.cell(row=curr_row, column=2).border = thin_border
        
        for c in range(3, 6):
            cell = ws1.cell(row=curr_row, column=c, value="")
            cell.border = thin_border
            
        tot_pv = ws1.cell(row=curr_row, column=6, value=f"=SUM(F{cf_start_row}:F{cf_end_row})")
        tot_pv.font = bold_font
        tot_pv.alignment = align_right
        tot_pv.number_format = fmt_currency
        tot_pv.border = thin_border
        
        # Fill in the Excel Discounted Price cell in the info table
        price_cell = ws1.cell(row=val_row, column=8, value=f"=F{curr_row}")
        price_cell.font = bold_font
        price_cell.alignment = align_center
        price_cell.number_format = fmt_currency
        price_cell.border = thin_border
        price_cell.fill = green_fill
        
        curr_row += 3 # space before next bond

    # ═════════════════════════════════════════════════════════════
    # TAB 2: Duration & Convexity Validation
    # ═════════════════════════════════════════════════════════════
    ws2 = wb.create_sheet(title="Duration & Convexity Validation")
    ws2.views.sheetView[0].showGridLines = True
    
    ws2.cell(row=2, column=2, value="Duration & Convexity Analytical Verification").font = title_font
    ws2.cell(row=3, column=2, value="Formulas mapping: Macaulay Duration, Modified Duration, Convexity, and DV01").font = italic_font
    
    curr_row = 5
    for bond_id in selected_bond_ids:
        bond_row = df_bonds[df_bonds['BondID'] == bond_id].iloc[0]
        
        ws2.cell(row=curr_row, column=2, value=f"Risk Measures: {bond_id}").font = section_font
        curr_row += 1
        
        # Metrics comparison table
        metrics = ["Macaulay Duration", "Modified Duration", "Convexity", "DV01 (per 100 Face)"]
        for col_idx, m_name in enumerate(metrics, start=2):
            cell = ws2.cell(row=curr_row, column=col_idx, value=m_name)
            cell.font = bold_font
            cell.fill = accent_fill
            cell.alignment = align_center
            cell.border = thin_border
            
        val_row = curr_row + 1
        for col_idx, m_name in enumerate(metrics, start=2):
            cell = ws2.cell(row=val_row, column=col_idx)
            cell.font = bold_font
            cell.alignment = align_center
            cell.border = thin_border
            cell.fill = green_fill
            if m_name == "Macaulay Duration":
                cell.value = f"=SUM(C{val_row + 3}:C{val_row + 3 + len(cf) - 1}) / F{val_row + 3 + len(cf)}"
                cell.number_format = fmt_number
            elif m_name == "Modified Duration":
                # Mod = Mac / (1 + YTM / Freq)
                # Need to read YTM & Freq from Sheet1!
                # Row index for this bond's info sheet: we can hardcode the sheet1 cell refs!
                # Bond 1: ZTFI-0001 info row is 6. Bond 2: ZTFI-0003 info row is 21. Bond 3: ZTFI-0005 info row is 36.
                # Let's map bond_id to sheet1 row index
                sh1_row = 6 + selected_bond_ids.index(bond_id) * 15
                cell.value = f"=B{val_row} / (1 + 'Bond Pricing Validation'!F{sh1_row} / 'Bond Pricing Validation'!D{sh1_row})"
                cell.number_format = fmt_number
            elif m_name == "Convexity":
                # C = SUM(t * (t + 1/m) * PV) / (P * (1 + y/m)^2)
                sh1_row = 6 + selected_bond_ids.index(bond_id) * 15
                cell.value = f"=SUM(D{val_row + 3}:D{val_row + 3 + len(cf) - 1}) / ('Bond Pricing Validation'!H{sh1_row} * (1 + 'Bond Pricing Validation'!F{sh1_row} / 'Bond Pricing Validation'!D{sh1_row})^2)"
                cell.number_format = fmt_number
            elif m_name == "DV01 (per 100 Face)":
                # DV01 = ModDuration * Price * 0.0001
                sh1_row = 6 + selected_bond_ids.index(bond_id) * 15
                cell.value = f"=C{val_row} * 'Bond Pricing Validation'!H{sh1_row} * 0.0001"
                cell.number_format = fmt_number
                
        # Schedule headers
        curr_row += 3
        headers = ["Period", "Time (t)", "t * PV (Macaulay)", "t * (t + 1/m) * PV (Convexity)"]
        for col_idx, h in enumerate(headers, start=2):
            cell = ws2.cell(row=curr_row, column=col_idx, value=h)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = align_center
            cell.border = thin_border
            
        cf, times = generate_bond_cashflows(
            bond_row['FaceValue'], bond_row['CouponRate'],
            int(bond_row['CouponFrequency']), bond_row['YearsToMaturity']
        )
        
        cf_start_row = curr_row + 1
        sh1_cf_start = 9 + selected_bond_ids.index(bond_id) * 15
        sh1_row = 6 + selected_bond_ids.index(bond_id) * 15
        
        for idx, t_val in enumerate(times):
            curr_row += 1
            sh1_cf_row = sh1_cf_start + idx
            
            # Period
            c_per = ws2.cell(row=curr_row, column=2, value=idx + 1)
            c_per.font = regular_font
            c_per.alignment = align_center
            c_per.border = thin_border
            # Time
            c_t = ws2.cell(row=curr_row, column=3, value=t_val)
            c_t.font = regular_font
            c_t.alignment = align_right
            c_t.number_format = fmt_number
            c_t.border = thin_border
            # t * PV (Formula: t * Sheet1!PV)
            c_tpv = ws2.cell(row=curr_row, column=4, value=f"=C{curr_row} * 'Bond Pricing Validation'!F{sh1_cf_row}")
            c_tpv.font = regular_font
            c_tpv.alignment = align_right
            c_tpv.number_format = fmt_currency
            c_tpv.border = thin_border
            # t * (t + 1/m) * PV (Formula: t * (t + 1 / Sheet1!Freq) * Sheet1!PV)
            c_t_t_m_pv = ws2.cell(row=curr_row, column=5, value=f"=C{curr_row} * (C{curr_row} + 1 / 'Bond Pricing Validation'!D{sh1_row}) * 'Bond Pricing Validation'!F{sh1_cf_row}")
            c_t_t_m_pv.font = regular_font
            c_t_t_m_pv.alignment = align_right
            c_t_t_m_pv.number_format = fmt_currency
            c_t_t_m_pv.border = thin_border
            
        cf_end_row = curr_row
        
        # Totals
        curr_row += 1
        ws2.cell(row=curr_row, column=2, value="Sum").font = bold_font
        ws2.cell(row=curr_row, column=2).alignment = align_left
        ws2.cell(row=curr_row, column=2).border = thin_border
        
        ws2.cell(row=curr_row, column=3, value="").border = thin_border
        
        tot_tpv = ws2.cell(row=curr_row, column=4, value=f"=SUM(D{cf_start_row}:D{cf_end_row})")
        tot_tpv.font = bold_font
        tot_tpv.alignment = align_right
        tot_tpv.number_format = fmt_currency
        tot_tpv.border = thin_border
        
        tot_ttmpv = ws2.cell(row=curr_row, column=5, value=f"=SUM(E{cf_start_row}:E{cf_end_row})")
        tot_ttmpv.font = bold_font
        tot_ttmpv.alignment = align_right
        tot_ttmpv.number_format = fmt_currency
        tot_ttmpv.border = thin_border
        
        curr_row += 3

    # ═════════════════════════════════════════════════════════════
    # TAB 3: Portfolio Aggregation Crosscheck
    # ═════════════════════════════════════════════════════════════
    ws3 = wb.create_sheet(title="Portfolio Crosscheck")
    ws3.views.sheetView[0].showGridLines = True
    
    ws3.cell(row=2, column=2, value="Portfolio Aggregation Crosscheck").font = title_font
    ws3.cell(row=3, column=2, value="Verifying portfolio weight-based Modified Duration, Convexity, and DV01").font = italic_font
    
    headers = [
        "Bond ID", "Face Value", "YTM", "Clean Price", "Dirty Price",
        "Market Value (Local)", "Market Value (INR)", "Portfolio Weight",
        "Modified Duration", "Weighted Duration", "Convexity", "Weighted Convexity",
        "DV01 (per 100 Face)", "Weighted DV01"
    ]
    
    # Table headers
    curr_row = 5
    for col_idx, h in enumerate(headers, start=2):
        cell = ws3.cell(row=curr_row, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = align_center
        cell.border = thin_border
        
    start_data_row = curr_row + 1
    
    # Write bond data
    for idx, row in df_bonds.iterrows():
        curr_row += 1
        r_fill = zebra_fill if idx % 2 == 1 else PatternFill(fill_type=None)
        
        # Bond ID
        c = ws3.cell(row=curr_row, column=2, value=row['BondID'])
        c.font = regular_font; c.alignment = align_center; c.border = thin_border; c.fill = r_fill
        # Face Value
        c = ws3.cell(row=curr_row, column=3, value=row['FaceValue'])
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.fill = r_fill; c.number_format = fmt_currency
        # YTM
        c = ws3.cell(row=curr_row, column=4, value=row['YieldToMaturity'])
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.fill = r_fill; c.number_format = fmt_percent
        # Clean Price
        c = ws3.cell(row=curr_row, column=5, value=row['CleanPrice'])
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.fill = r_fill; c.number_format = fmt_currency
        # Dirty Price
        c = ws3.cell(row=curr_row, column=6, value=row['DirtyPrice'])
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.fill = r_fill; c.number_format = fmt_currency
        # MV Local
        c = ws3.cell(row=curr_row, column=7, value=row['MarketValue_LocalCcy'])
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.fill = r_fill; c.number_format = fmt_currency
        # MV INR
        c = ws3.cell(row=curr_row, column=8, value=row['MarketValue_INR'])
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.fill = r_fill; c.number_format = fmt_currency
        
        # Weight Formula: MV_INR / SUM(MV_INR)
        # Note: Sum is at row curr_row + 1 (totals row written next)
        c = ws3.cell(row=curr_row, column=9, value=f"=H{curr_row} / $H${len(df_bonds) + start_data_row}")
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.fill = r_fill; c.number_format = fmt_percent
        
        # Mod Duration
        c = ws3.cell(row=curr_row, column=10, value=row['ModifiedDuration'])
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.fill = r_fill; c.number_format = fmt_number
        
        # Weighted Duration: Weight * ModDuration
        c = ws3.cell(row=curr_row, column=11, value=f"=I{curr_row} * J{curr_row}")
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.fill = r_fill; c.number_format = fmt_number
        
        # Convexity
        c = ws3.cell(row=curr_row, column=12, value=row['Convexity'])
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.fill = r_fill; c.number_format = fmt_number
        
        # Weighted Convexity: Weight * Convexity
        c = ws3.cell(row=curr_row, column=13, value=f"=I{curr_row} * L{curr_row}")
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.fill = r_fill; c.number_format = fmt_number
        
        # DV01 (per 100 face)
        c = ws3.cell(row=curr_row, column=14, value=row['DV01_Per100Face'])
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.fill = r_fill; c.number_format = fmt_number
        
        # Weighted DV01: Weight * DV01
        c = ws3.cell(row=curr_row, column=15, value=f"=I{curr_row} * N{curr_row}")
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.fill = r_fill; c.number_format = fmt_number
        
    end_data_row = curr_row
    
    # Totals Row
    curr_row += 1
    ws3.cell(row=curr_row, column=2, value="Portfolio Totals").font = bold_font
    ws3.cell(row=curr_row, column=2).alignment = align_left
    ws3.cell(row=curr_row, column=2).border = double_bottom_border
    ws3.cell(row=curr_row, column=2).fill = accent_fill
    
    for c_idx in range(3, 7):
        cell = ws3.cell(row=curr_row, column=c_idx, value="")
        cell.border = double_bottom_border
        cell.fill = accent_fill
        
    # Total Market Value INR
    tot_mv = ws3.cell(row=curr_row, column=8, value=f"=SUM(H{start_data_row}:H{end_data_row})")
    tot_mv.font = bold_font; tot_mv.alignment = align_right; tot_mv.number_format = fmt_currency; tot_mv.border = double_bottom_border; tot_mv.fill = accent_fill
    
    # Total Weight (should sum to 1.000)
    tot_wt = ws3.cell(row=curr_row, column=9, value=f"=SUM(I{start_data_row}:I{end_data_row})")
    tot_wt.font = bold_font; tot_wt.alignment = align_right; tot_wt.number_format = fmt_percent; tot_wt.border = double_bottom_border; tot_wt.fill = accent_fill
    
    # Empty
    ws3.cell(row=curr_row, column=10, value="").border = double_bottom_border; ws3.cell(row=curr_row, column=10).fill = accent_fill
    
    # Portfolio Modified Duration
    port_dur = ws3.cell(row=curr_row, column=11, value=f"=SUM(K{start_data_row}:K{end_data_row})")
    port_dur.font = bold_font; port_dur.alignment = align_right; port_dur.number_format = fmt_number; port_dur.border = double_bottom_border; port_dur.fill = green_fill
    
    # Empty
    ws3.cell(row=curr_row, column=12, value="").border = double_bottom_border; ws3.cell(row=curr_row, column=12).fill = accent_fill
    
    # Portfolio Convexity
    port_conv = ws3.cell(row=curr_row, column=13, value=f"=SUM(M{start_data_row}:M{end_data_row})")
    port_conv.font = bold_font; port_conv.alignment = align_right; port_conv.number_format = fmt_number; port_conv.border = double_bottom_border; port_conv.fill = green_fill
    
    # Empty
    ws3.cell(row=curr_row, column=14, value="").border = double_bottom_border; ws3.cell(row=curr_row, column=14).fill = accent_fill
    
    # Portfolio DV01
    port_dv01 = ws3.cell(row=curr_row, column=15, value=f"=SUM(O{start_data_row}:O{end_data_row})")
    port_dv01.font = bold_font; port_dv01.alignment = align_right; port_dv01.number_format = fmt_number; port_dv01.border = double_bottom_border; port_dv01.fill = green_fill

    # ═════════════════════════════════════════════════════════════
    # TAB 4: VaR Validation (Parametric vs Monte Carlo)
    # ═════════════════════════════════════════════════════════════
    ws4 = wb.create_sheet(title="VaR Backtesting Validation")
    ws4.views.sheetView[0].showGridLines = True
    
    ws4.cell(row=2, column=2, value="Value at Risk & Backtesting Validation").font = title_font
    ws4.cell(row=3, column=2, value="Parametric VaR vs Historical (Percentile) VaR with Kupiec POF Adequacy Test").font = italic_font
    
    # Summary of simulated PnL
    ws4.cell(row=5, column=2, value="PnL Series Statistics").font = section_font
    stats_labels = ["Mean P&L", "Std Dev P&L", "Observation Count (N)"]
    stats_formulas = [
        "=AVERAGE(I6:I1005)",
        "=STDEV.S(I6:I1005)",
        "=COUNT(I6:I1005)"
    ]
    
    for i, (lbl, fmla) in enumerate(zip(stats_labels, stats_formulas)):
        r = 6 + i
        ws4.cell(row=r, column=2, value=lbl).font = bold_font
        cell = ws4.cell(row=r, column=3, value=fmla)
        cell.font = regular_font
        cell.alignment = align_right
        cell.border = thin_border
        if "Count" in lbl:
            cell.number_format = fmt_integer
        else:
            cell.number_format = fmt_currency

    # VaR Method Comparison Table
    ws4.cell(row=10, column=2, value="VaR Comparison").font = section_font
    headers = ["Confidence Level", "Z-Score", "Parametric VaR", "Historical VaR", "Difference %"]
    for col_idx, h in enumerate(headers, start=2):
        cell = ws4.cell(row=11, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = align_center
        cell.border = thin_border
        
    conf_levels = [0.90, 0.95, 0.99]
    for idx, cl in enumerate(conf_levels):
        r = 12 + idx
        # Confidence
        cell = ws4.cell(row=r, column=2, value=cl)
        cell.font = regular_font; cell.alignment = align_center; cell.border = thin_border; cell.number_format = fmt_percent
        # Z-Score
        cell = ws4.cell(row=r, column=3, value=f"=NORM.S.INV(1 - B{r})")
        cell.font = regular_font; cell.alignment = align_right; cell.border = thin_border; cell.number_format = fmt_number
        # Parametric VaR: -(Mean + Z * StdDev) -> -$C$6 - C12 * $C$7
        cell = ws4.cell(row=r, column=4, value=f"=-($C$6 + C{r} * $C$7)")
        cell.font = regular_font; cell.alignment = align_right; cell.border = thin_border; cell.number_format = fmt_currency
        # Historical VaR
        cell = ws4.cell(row=r, column=5, value=f"=-PERCENTILE.INC(I6:I1005, 1 - B{r})")
        cell.font = regular_font; cell.alignment = align_right; cell.border = thin_border; cell.number_format = fmt_currency
        # Difference %
        cell = ws4.cell(row=r, column=6, value=f"=ABS(D{r} - E{r}) / E{r}")
        cell.font = regular_font; cell.alignment = align_right; cell.border = thin_border; cell.number_format = fmt_percent

    # Kupiec POF Test Table
    ws4.cell(row=16, column=2, value="Kupiec POF Backtesting Adequacy Test (99% VaR)").font = section_font
    kupiec_labels = [
        "Expected Violation Rate (p0)",
        "Observations (N)",
        "Violations (Breaches)",
        "Observed Violation Rate (p)",
        "Likelihood Ratio (LR) Statistic",
        "LR Chi-Sq Critical Value (5% sig)",
        "Kupiec P-Value",
        "Adequacy Result"
    ]
    
    # We will write formulas referencing the data
    kupiec_formulas = [
        "=1 - 0.99",
        "=$C$8",
        '=COUNTIF(I6:I1005, "<" & -E14)', # Breaches of Historical 99% VaR
        "=C19 / C18",
        '=IF(C19=0, -2 * C18 * LN(1 - C17), -2 * ((C18 - C19) * LN(1 - C17) + C19 * LN(C17) - (C18 - C19) * LN(1 - C20) - C19 * LN(C20)))',
        '=CHISQ.INV(0.95, 1)',
        '=1 - CHISQ.DIST(C21, 1, TRUE)',
        '=IF(C21 < C22, "PASS ✅", "FAIL ❌")'
    ]
    
    for i, (lbl, fmla) in enumerate(zip(kupiec_labels, kupiec_formulas)):
        r = 17 + i
        ws4.cell(row=r, column=2, value=lbl).font = bold_font
        cell = ws4.cell(row=r, column=3, value=fmla)
        cell.font = bold_font if lbl == "Adequacy Result" else regular_font
        cell.alignment = align_center if lbl == "Adequacy Result" else align_right
        cell.border = thin_border
        
        if lbl in ["Expected Violation Rate (p0)", "Observed Violation Rate (p)", "Kupiec P-Value"]:
            cell.number_format = fmt_percent
        elif lbl == "Observations (N)" or lbl == "Violations (Breaches)":
            cell.number_format = fmt_integer
        elif lbl in ["Likelihood Ratio (LR) Statistic", "LR Chi-Sq Critical Value (5% sig)"]:
            cell.number_format = fmt_number
        elif lbl == "Adequacy Result":
            cell.fill = green_fill

    # Write MC PnL data in columns H and I for formulas to reference
    ws4.cell(row=5, column=8, value="Scenario ID").font = bold_font
    ws4.cell(row=5, column=9, value="PnL Total INR").font = bold_font
    ws4.cell(row=5, column=8).fill = accent_fill
    ws4.cell(row=5, column=9).fill = accent_fill
    
    for idx, row in df_mc.iterrows():
        r = 6 + idx
        c_id = ws4.cell(row=r, column=8, value=row.get('ScenarioID', f"SC-{idx+1:04d}"))
        c_id.font = regular_font; c_id.alignment = align_center; c_id.border = thin_border
        
        c_pnl = ws4.cell(row=r, column=9, value=row['PnL_Total_INR'])
        c_pnl.font = regular_font; c_pnl.alignment = align_right; c_pnl.border = thin_border; c_pnl.number_format = fmt_currency

    # ═════════════════════════════════════════════════════════════
    # TAB 5: Nelson-Siegel fitted curve vs actual
    # ═════════════════════════════════════════════════════════════
    ws5 = wb.create_sheet(title="Yield Curve Fitting")
    ws5.views.sheetView[0].showGridLines = True
    
    ws5.cell(row=2, column=2, value="Nelson-Siegel-Svensson Fitted Curve vs Actual").font = title_font
    ws5.cell(row=3, column=2, value="Evaluating mathematical yield curve predictions vs market observed yields").font = italic_font
    
    # Fit NSS to latest yield curve to get parameters
    latest_date = df_yc['CurveDate'].max()
    latest_yc = df_yc[df_yc['CurveDate'] == latest_date].sort_values('Tenor_Years')
    
    nss = NelsonSiegelSvensson()
    nss.fit(latest_yc['Tenor_Years'].values, latest_yc['Yield'].values)
    
    # Write NSS Parameters
    ws5.cell(row=5, column=2, value="NSS Parameters").font = section_font
    param_names = ["Beta 0 (Level)", "Beta 1 (Slope)", "Beta 2 (Curvature 1)", "Beta 3 (Curvature 2)", "Tau 1 (Decay 1)", "Tau 2 (Decay 2)"]
    param_vals = nss.params
    
    for i, (name, val) in enumerate(zip(param_names, param_vals)):
        r = 6 + i
        ws5.cell(row=r, column=2, value=name).font = bold_font
        cell = ws5.cell(row=r, column=3, value=val)
        cell.font = regular_font; cell.alignment = align_right; cell.border = thin_border; cell.number_format = fmt_number
        
    # Table of curve prediction
    ws5.cell(row=14, column=2, value="Tenor Fitting Detail").font = section_font
    headers = ["Tenor (Years)", "Actual Yield", "Excel Fitted NSS Yield", "Residual", "Squared Residual"]
    for col_idx, h in enumerate(headers, start=2):
        cell = ws5.cell(row=15, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = align_center
        cell.border = thin_border
        
    start_curve_row = 16
    for idx, row in latest_yc.iterrows():
        r = start_curve_row + idx
        # Tenor
        c = ws5.cell(row=r, column=2, value=row['Tenor_Years'])
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.number_format = fmt_number
        # Actual Yield
        c = ws5.cell(row=r, column=3, value=row['Yield'])
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.number_format = fmt_percent
        
        # Excel NSS Formula:
        # NSS = Beta0 + Beta1 * (1 - EXP(-t/Tau1))/(t/Tau1) + Beta2 * ((1 - EXP(-t/Tau1))/(t/Tau1) - EXP(-t/Tau1)) + Beta3 * ((1 - EXP(-t/Tau2))/(t/Tau2) - EXP(-t/Tau2))
        # Refs: Beta0=$C$6, Beta1=$C$7, Beta2=$C$8, Beta3=$C$9, Tau1=$C$10, Tau2=$C$11
        # t = B{r}
        t_ref = f"B{r}"
        formula_nss = (
            f"=$C$6 + $C$7 * (1 - EXP(-{t_ref} / $C$10)) / ({t_ref} / $C$10) + "
            f"$C$8 * ((1 - EXP(-{t_ref} / $C$10)) / ({t_ref} / $C$10) - EXP(-{t_ref} / $C$10)) + "
            f"$C$9 * ((1 - EXP(-{t_ref} / $C$11)) / ({t_ref} / $C$11) - EXP(-{t_ref} / $C$11))"
        )
        c = ws5.cell(row=r, column=4, value=formula_nss)
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.number_format = fmt_percent
        
        # Residual = Actual - Fitted
        c = ws5.cell(row=r, column=5, value=f"=C{r} - D{r}")
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.number_format = fmt_percent
        
        # Squared Residual
        c = ws5.cell(row=r, column=6, value=f"=E{r} ^ 2")
        c.font = regular_font; c.alignment = align_right; c.border = thin_border; c.number_format = fmt_number
        
    end_curve_row = start_curve_row + len(latest_yc) - 1
    
    # RMSE Summary Row
    curr_row = end_curve_row + 1
    ws5.cell(row=curr_row, column=2, value="RMSE").font = bold_font
    ws5.cell(row=curr_row, column=2).alignment = align_left
    ws5.cell(row=curr_row, column=2).border = double_bottom_border
    ws5.cell(row=curr_row, column=2).fill = accent_fill
    
    for c_idx in range(3, 6):
        cell = ws5.cell(row=curr_row, column=c_idx, value="")
        cell.border = double_bottom_border
        cell.fill = accent_fill
        
    # RMSE Formula: SQRT(SUM(Sq_Residuals) / COUNT(Tenors))
    rmse_cell = ws5.cell(row=curr_row, column=6, value=f"=SQRT(SUM(F{start_curve_row}:F{end_curve_row}) / COUNT(B{start_curve_row}:B{end_curve_row}))")
    rmse_cell.font = bold_font; rmse_cell.alignment = align_right; rmse_cell.number_format = fmt_number; rmse_cell.border = double_bottom_border; rmse_cell.fill = green_fill

    # ═════════════════════════════════════════════════════════════
    # Set column widths automatically for all sheets
    # ═════════════════════════════════════════════════════════════
    for ws in wb.worksheets:
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                # Avoid using formula string for length calculation
                if cell.value:
                    val_str = str(cell.value)
                    if not val_str.startswith("="):
                        max_len = max(max_len, len(val_str))
                    else:
                        max_len = max(max_len, 15) # default for formulas
            ws.column_dimensions[col_letter].width = max(max_len + 3, 12)
            
    # Save file
    file_path = OUTPUT_DIR / "validation_workbook.xlsx"
    wb.save(file_path)
    print(f"Validation workbook saved successfully to {file_path} ✅")

if __name__ == "__main__":
    create_validation_workbook()
