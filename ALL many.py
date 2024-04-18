import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import lmfit

# XML 파일 경로
xml_file = '../HY202103_D07_(0,0)_LION1_DCM_LMZC.xml'

# XML 파일 파싱
tree = ET.parse(xml_file)
root = tree.getroot()

# IV 데이터를 그래프에 그리는 함수
def plot_iv_data(ax):
    # IVMeasurement 요소 찾기
    iv_measurement_element = root.find('.//IVMeasurement')

    # Voltage 및 Current 요소의 텍스트 값을 파싱하여 출력
    voltage_text = iv_measurement_element.find('.//Voltage').text
    current_text = iv_measurement_element.find('.//Current').text

    # Voltage 및 Current 텍스트 값을 파싱하여 실수형 리스트로 변환
    voltage_values = [float(value) for value in voltage_text.split(',')]
    current_values = [float(value) for value in current_text.split(',')]

    # 적합 실행 (알고리즘 변경)
    def diode_equation(V, Is, n, Vt, V_linear, Ilinear):
        current = []
        for v in V:
            if v >= V_linear:
                current.append(Is * (np.exp(v / (n * Vt)) - 1))
            else:
                current.append(Ilinear * v)
        return current

    # 초기 추정값 설정
    Is_guess = current_values[0]
    n_guess = 1.0
    Vt_guess = 0.0256
    Ilinear_guess = 0.0
    Vlinear_guess = 0.0

    # 매개변수 및 초기 추정값 정의
    params = lmfit.Parameters()
    params.add('Is', value=Is_guess, min=0)  # 포화 전류
    params.add('n', value=n_guess, min=1)  # 이상성 지수
    params.add('Vt', value=Vt_guess, min=0)  # 열전압
    params.add('Ilinear', value=Ilinear_guess)  # 음수 전압 영역에서의 전류
    params.add('V_linear', value=Vlinear_guess)  # 음수 전압 영역에서의 선형 근사 전압

    # 적합 실행 (알고리즘 변경)
    result = lmfit.minimize(
        # 잔차 함수
        lambda params, x, y: np.array(diode_equation(x, **params)) - np.array(y),
        # 매개변수 및 데이터
        params, args=(voltage_values, current_values),
        # 알고리즘 변경
        method='least squares'
    )

    # 적합된 값 얻기
    best_fit = np.abs(current_values) + result.residual

    # R-squared 값 계산
    ss_residual = np.sum(result.residual ** 2)
    ss_total = np.sum(np.abs(current_values) - np.abs(np.mean(current_values)) ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    # 데이터를 그래프에 그리기
    ax.scatter(voltage_values, np.abs(current_values), label='Original Data')
    ax.plot(voltage_values, best_fit, color='red',
            label=f'Fitted Polynomial \n R-squared: {round(r_squared, 3)}')
    ax.set_yscale('log')  # y축 로그 스케일로 변경
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('Absolute Current [A]')
    ax.set_title('IV raw dat & fitted dat')
    ax.legend()


def plot_transmission_spectra_all(ax):

    # 그래프에 그릴 데이터를 담을 리스트 초기화
    data_to_plot = []

    WavelengthSweep = list(root.findall('.//WavelengthSweep'))



    # 모든 WavelengthSweep 요소 반복
    for WavelengthSweep in root.findall('.//WavelengthSweep'):
        # DCBias 속성 값 가져오기
        dc_bias = float(WavelengthSweep.get('DCBias'))

        # LengthUnit과 transmission 요소의 text 값 가져오기
        length_values = []
        measured_transmission_values = []
        for L in WavelengthSweep.findall('.//L'):
            length_text = L.text
            length_text = length_text.replace(',', ' ')
            length_values.extend([float(value) for value in length_text.split() if value.strip()])

        for IL in WavelengthSweep.findall('.//IL'):
            measured_transmission_text = IL.text
            measured_transmission_text = measured_transmission_text.replace(',', ' ')
            measured_transmission_values.extend(
                [float(value) for value in measured_transmission_text.split() if value.strip()])

        # 데이터를 데이터 플롯 리스트에 추가
        data_to_plot.append((dc_bias, length_values, measured_transmission_values))

    # 그래프 그리기
    for dc_bias, length_values, measured_transmission_values in data_to_plot:
        plt.plot(length_values, measured_transmission_values, label=f'DCBias={dc_bias}V')

    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Measured Transmission [dB]')
    ax.set_title(f'Transmission Spectra - as measured')
    ax.legend(title='DC Bias', loc='upper right')
    ax.grid(True)


# 데이터를 그래프에 그리는 함수
def plot_transmission_spectra(ax):
    # 여섯 번째 WavelengthSweep 요소 선택
    WavelengthSweep = list(root.findall('.//WavelengthSweep'))[6]

    # LengthUnit과 transmission 요소의 text 값 가져오기
    length_values = []
    measured_transmission_values = []
    for L in WavelengthSweep.findall('.//L'):
        length_text = L.text
        length_text = length_text.replace(',', ' ')
        length_values.extend([float(value) for value in length_text.split() if value.strip()])

    for IL in WavelengthSweep.findall('.//IL'):
        measured_transmission_text = IL.text
        measured_transmission_text = measured_transmission_text.replace(',', ' ')
        measured_transmission_values.extend(
            [float(value) for value in measured_transmission_text.split() if value.strip()])

    # 원래 데이터를 검은색 점으로 그리기
    ax.scatter(length_values, measured_transmission_values, color='black', label='Measured Data')

    # 다항식 차수 범위 설정
    poly_degrees = range(1, 14)

    # 각 차수에 대한 fitting 결과 저장할 리스트 초기화
    fitting_results = []

    # 1차부터 13차까지의 fitting 결과 저장
    for degree in range(1, 14):
        coeffs = np.polyfit(length_values, measured_transmission_values, degree)
        p = np.poly1d(coeffs)
        yhat = p(length_values)
        ybar = np.sum(measured_transmission_values) / len(measured_transmission_values)
        ssreg = np.sum((yhat - ybar) ** 2)
        sstot = np.sum((measured_transmission_values - ybar) ** 2)
        r_squared = ssreg / sstot
        fitting_results.append((coeffs, r_squared))



    # 각 차수에 대한 fitting 그래프 그리기
    for degree, coeffs in enumerate(fitting_results, start=1):
        p = np.poly1d(coeffs[0])
        x_values = np.linspace(min(length_values), max(length_values), 100)
        y_values = p(x_values)
        ax.plot(x_values, y_values, label=f'{degree} Degree Fit')


    # 근사식과 R 제곱 값 출력
    best_degree = np.argmax([result[1] for result in fitting_results]) + 1
    best_coeffs = fitting_results[best_degree - 1][0]
    equation = ' + '.join([f'{round(coeff, 2)}*x^{best_degree - i}' for i, coeff in enumerate(best_coeffs[::-1])])
    plt.text(1540, -13, f'Fitted Equation:\n{equation}', fontsize=12, color='red')
    print(equation)

    r_squared_text = f'R^2: {round(fitting_results[best_degree - 1][1], 3)}, Degree: {best_degree}'
    plt.text(1540, -14, r_squared_text, fontsize=12, color='red')


    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Measured Transmission [dB]')
    ax.set_title(f'Transmission Spectra - Processed and fitting')
    ax.legend(loc='upper right', bbox_to_anchor=(1.6, 1.1))
    ax.grid(True)



# 데이터를 그래프에 그리기 위한 subplot 생성
plt.figure(figsize=(15, 10))

# 첫 번째 서브플롯에 transmission spectra 그리기
plt.subplot(2,3,1)
plot_transmission_spectra_all(plt.gca())

# 두 번째 서브플롯에 transmission spectra ref fitting 그리기
plt.subplot(2,3,2)
plot_transmission_spectra(plt.gca())

# 네 번째 서브플롯에 iv curve 그리기
plt.subplot(2,3,4)
plot_iv_data(plt.gca())
plt.subplots_adjust(hspace=0.3) # 위아래 간격 안 맞아서 수동 조정
plt.show()

