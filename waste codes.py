# from evaluation

if len(extrema_upper_index) + len(extrema_lower_index) <= 0:  # if there is no real extrema
    distance = last_extrema  # means that there is no enough extremas to do the calculation
    amplitude_upper_ema = max(imfs[n])
    amplitude_lower_ema = min(imfs[n])
    step = abs(amplitude_upper_ema - amplitude_lower_ema) / distance
    reference_amplitude = abs(imfs[n][-1]) + 2 * step
elif len(extrema_upper_index) + len(extrema_lower_index) == 1:  # if there is only one extrema
    distance = len(imfs[n]) - last_extrema
    amplitude_upper_ema = max(imfs[n][last_extrema], imfs[n][-1])
    amplitude_lower_ema = min(imfs[n][last_extrema], imfs[n][-1])
    step = abs(amplitude_upper_ema - amplitude_lower_ema) / distance
    reference_amplitude = abs(imfs[n][-1]) + 2 * step
else:
    amplitude_upper_ema = ema(extrema_upper_value, alpha=0.6)
    amplitude_lower_ema = ema(extrema_lower_value, alpha=0.6)
    nextremas = min(len(extrema_lower_index), len(extrema_upper_index))
    distance_set = abs(extrema_upper_index[-nextremas:] - extrema_lower_index[-nextremas:])
    distance = ema(distance_set, alpha=0.6)
    step = abs(amplitude_upper_ema - amplitude_lower_ema) / distance
    reference_amplitude = abs(amplitude_lower_ema) * 0.25 + abs(amplitude_upper_ema) * 0.25 + abs(
        imfs[n][last_extrema]) * 0.5
# do the rough forecast from here:
if n >= np.floor(nimfs / 2):
    step = abs(imfs[n][-1] - imfs[n][-2])

if imfs[n][last_extrema] * imfs[n][-1] < 0:  # if the last point has already crossed the axis
    if distance <= 1.58:  # have to switch the direction
        forecast_value = imfs[n][-1] - imfs[n][-1] / abs(imfs[n][-1]) * step
    else:
        if abs(imfs[n][-1]) + step > reference_amplitude:
            forecast_value = imfs[n][-1] / abs(imfs[n][-1]) * reference_amplitude
        else:
            forecast_value = imfs[n][-1] / abs(imfs[n][-1]) * (abs(imfs[n][-1]) + step)
elif imfs[n][-1] - imfs[n][-2] == 0:
    forecast_value = 0  # give up the forecast
else:
    if distance < 1.1:  # means have a more often switch
        forecast_value = imfs[n][-1] - step * (imfs[n][-1] - imfs[n][-2]) / abs(
            (imfs[n][-1] - imfs[n][-2]))  # change the direction
    else:
        forecast_value = imfs[n][-1] + step * (imfs[n][-1] - imfs[n][-2]) / abs(
            (imfs[n][-1] - imfs[n][-2]))  # continue with the trend