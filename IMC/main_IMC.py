
from create_stos_IMC import run_create_stos_IMC


create_stos = True
if create_stos:

    # Choose which subjects to process
    subject_code_list = ['P001', 'P002']
    # subject_code_list = [f'P{str(i).zfill(3)}' for i in range(1, 21)]   # Or run the function for all subjects

    for subject_code in subject_code_list:

        run_create_stos_IMC(subject_code, test=False)
