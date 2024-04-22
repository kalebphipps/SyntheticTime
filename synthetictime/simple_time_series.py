from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


class SimpleTimeSeries:
    """
    This models a simple synthetic time series.

    Attributes
    ----------
    size : int
        The size of the time series.
    base_amplitude: float
        The amplitude of the sine curve for the base time series.
    base_frequency: float
        The frequency of the sine curve for the base time series.
    base_noise_scale: float
        The scale of the base noise.
    base_noise_amplitude: float
        The amplitude of the base noise.
    number_of_trend_events: Optional[int]
        The number of trend events to include in the time series.
    trend_slope_low: Optional[float]
        The lower boundary for sampling the slope of the trend events.
    trend_slope_high: Optional[float]
        The upper boundary for sampling the slope of the trend events.
    number_of_cosine_events: Optional[int]
        The number of cosine events.
    cosine_frequency_low: Optional[float]
        The lower boundary for sampling the frequency of the cosine events.
    cosine_frequency_high: Optional[float]
        The upper boundary for sampling the frequency of the cosine events.
    cosine_amplitude_low: Optional[float]
        The lower boundary for sampling the amplitude of the cosine events.
    cosine_amplitude_high: Optional[float]
        The upper boundary for sampling the amplitude of the cosine events.
    number_of_increased_noise_events: Optional[int]
        The number of increased noise events.
    increased_noise_low: Optional[float]
        The lower boundary for sampling the increased noise events.
    increased_noise_high: Optional[float]
        The upper boundary for sampling the increased noise.
    noise_type: Optional[str]
        The type of noise, currently only additive is possible.
    event_length_lower: Optional[float]
        The lower boundary for sampling the length of the events.
    event_length_upper: Optional[float]
        The upper boundary for sampling the length of the events.
    synthetic_time_series : np.ndarray
        The final synthetic time series generated.
    base_time_series : np.ndarray
        The base component of the generated time series.
    trend_time_series : np.ndarray
        The trend component of the generated time series.
    self.cosine_time_series : np.ndarray
        The cosine component of the generated time series.
    self.increased_noise_time_series : np.ndarray
        The increased noise component of the generated time series.

    Methods
    -------
    generate_time_series()
        Generates the time series based on the defined parameters.
    plot_time_series()
        Plots the generated time series and its components.
    """

    def __init__(
        self,
        size: int,
        base_amplitude: float,
        base_frequency: float,
        base_noise_scale: float,
        base_noise_amplitude: float,
        number_of_trend_events: Optional[int] = 0,
        trend_slope_low: Optional[float] = 0.01,
        trend_slope_high: Optional[float] = 0.2,
        number_of_cosine_events: Optional[int] = 0,
        cosine_frequency_low: Optional[float] = 0.01,
        cosine_frequency_high: Optional[float] = 0.2,
        cosine_amplitude_low: Optional[float] = 2.0,
        cosine_amplitude_high: Optional[float] = 8.0,
        number_of_increased_noise_events: Optional[int] = 0,
        increased_noise_low: Optional[float] = 0.8,
        increased_noise_high: Optional[float] = 2,
        noise_type: Optional[str] = "additive",
        event_length_lower: Optional[float] = None,
        event_length_upper: Optional[float] = None,
    ) -> None:
        """
        Initialize the simple time series with the defined parameters.

        Parameters
        ----------
        size : int
            The size of the time series.
        base_amplitude: float
            The amplitude of the sine curve for the base time series.
        base_frequency: float
            The frequency of the sine curve for the base time series.
        base_noise_scale: float
            The scale of the base noise.
        base_noise_amplitude: float
            The amplitude of the base noise.
        number_of_trend_events: Optional[int]
            The number of trend events to include in the time series.
        trend_slope_low: Optional[float]
            The lower boundary for sampling the slope of the trend events.
        trend_slope_high: Optional[float]
            The upper boundary for sampling the slope of the trend events.
        number_of_cosine_events: Optional[int]
            The number of cosine events.
        cosine_frequency_low: Optional[float]
            The lower boundary for sampling the frequency of the cosine events.
        cosine_frequency_high: Optional[float]
            The upper boundary for sampling the frequency of the cosine events.
        cosine_amplitude_low: Optional[float]
            The lower boundary for sampling the amplitude of the cosine events.
        cosine_amplitude_high: Optional[float]
            The upper boundary for sampling the amplitude of the cosine events.
        number_of_increased_noise_events: Optional[int]
            The number of increased noise events.
        increased_noise_low: Optional[float]
            The lower boundary for sampling the increased noise events.
        increased_noise_high: Optional[float]
            The upper boundary for sampling the increased noise.
        noise_type: Optional[str]
            The type of noise, currently only additive is possible.
        event_length_lower: Optional[float]
            The lower boundary for sampling the length of the events.
        event_length_upper: Optional[float]
            The upper boundary for sampling the length of the events.
        """
        self.size = size
        self.base_amplitude = base_amplitude
        self.base_frequency = base_frequency
        self.base_noise_scale = base_noise_scale
        self.base_noise_amplitude = base_noise_amplitude
        self.number_of_trend_events = number_of_trend_events
        self.trend_slope_low = trend_slope_low
        self.trend_slope_high = trend_slope_high
        self.number_of_cosine_events = number_of_cosine_events
        self.cosine_frequency_low = cosine_frequency_low
        self.cosine_frequency_high = cosine_frequency_high
        self.cosine_amplitude_low = cosine_amplitude_low
        self.cosine_amplitude_high = cosine_amplitude_high
        self.number_of_increased_noise_events = number_of_increased_noise_events
        self.increased_noise_low = increased_noise_low
        self.increased_noise_high = increased_noise_high
        self.noise_type = noise_type
        self.event_length_lower = event_length_lower
        self.event_length_upper = event_length_upper
        (
            self.synthetic_time_series,
            self.base_time_series,
            self.trend_time_series,
            self.cosine_time_series,
            self.increased_noise_time_series,
        ) = self.generate_time_series()

    def generate_time_series(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        time_index = np.linspace(0, int(self.size), int(self.size))
        """
        Generate the synthetic time series given the defined parameters.
        
        Return
        -----
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            The synthetic time series and its individual components including base time series, trend event time series, 
            cosine event time series, and increased noise event time series.
        """
        # Create boundaries for even lengths
        if self.event_length_lower is None:
            self.event_length_lower = int(int(self.size) / 50)
        if self.event_length_upper is None:
            self.event_length_upper = int(int(self.size) / 50)

        # Generate base synthetic time series as sine curve with noise
        if self.noise_type == "additive":
            synthetic_time_series = self.base_amplitude * np.sin(
                self.base_frequency * 2 * np.pi * time_index
            ) + self.base_noise_amplitude * np.random.normal(
                loc=0, scale=self.base_noise_scale, size=self.size
            )
        else:
            raise NotImplementedError(
                "Synthetic time series currently only implemented for additive noise!"
            )

        # Introduce occasional trend events into the time series
        trend_event_indices = np.random.choice(
            np.arange(int(self.size)), size=self.number_of_trend_events, replace=False
        )
        trend_time_series = np.zeros(len(time_index))
        for idx in trend_event_indices:
            trend_event_length = int(
                np.random.uniform(
                    low=self.event_length_lower, high=self.event_length_upper
                )
            )
            trend_slope = np.random.uniform(
                low=self.trend_slope_low, high=self.trend_slope_high
            )
            # For trend events occurring within the time series
            if idx + trend_event_length < len(time_index):
                multipliers = np.arange(start=0, stop=trend_event_length, step=1)
                trend_time_series[idx : idx + trend_event_length] = (
                    trend_slope * multipliers
                )
            # For trend events occurring at the end of the time series
            else:
                remainder = len(time_index) - idx
                multipliers = np.arange(start=0, stop=remainder, step=1)
                trend_time_series[idx:] = trend_slope * multipliers

        # Introduce occasional cosine events
        cosine_event_indices = np.random.choice(
            np.arange(int(self.size)), size=self.number_of_cosine_events, replace=False
        )
        cosine_time_series = np.zeros(len(time_index))
        for idx in cosine_event_indices:
            cosine_event_length = int(
                np.random.uniform(
                    low=self.event_length_lower, high=self.event_length_upper
                )
            )
            cosine_freq = np.random.uniform(
                low=self.cosine_frequency_low, high=self.cosine_frequency_high
            )
            cosine_amplitude = np.random.uniform(
                low=self.cosine_amplitude_low, high=self.cosine_amplitude_high
            )
            # For cosine events occurring within the time series
            if idx + cosine_event_length < len(time_index):
                cosine_time_series[idx : idx + cosine_event_length] = (
                    cosine_amplitude
                    * np.cos(
                        cosine_freq
                        * 2
                        * np.pi
                        * time_index[idx : idx + cosine_event_length]
                    )
                )
            # For cosine events occurring at the end of the time series
            else:
                cosine_time_series[idx:] = cosine_amplitude * np.cos(
                    cosine_freq * time_index[idx:]
                )

        # Introduce occasional increased noise events
        increased_noise_event_indices = np.random.choice(
            np.arange(int(self.size)),
            size=self.number_of_increased_noise_events,
            replace=False,
        )
        increased_noise_time_series = np.zeros(len(time_index))
        for idx in increased_noise_event_indices:
            increased_noise_event_length = int(
                np.random.uniform(
                    low=self.event_length_lower, high=self.event_length_upper
                )
            )
            random_scale = np.random.uniform(
                low=self.increased_noise_low, high=self.increased_noise_high
            )
            # For increased noise events occurring within the time series
            if idx + increased_noise_event_length < len(time_index):
                increased_noise_time_series[
                    idx : idx + increased_noise_event_length
                ] = np.random.normal(
                    loc=0, scale=random_scale, size=increased_noise_event_length
                )
            # For increased noise events occurring at the end of the time series
            else:
                remainder = len(time_index) - idx
                increased_noise_time_series[idx:] = np.random.normal(
                    loc=0, scale=random_scale, size=remainder
                )

        # Build the final time series as a combination of all time series components
        final_synthetic_time_series = (
            synthetic_time_series
            + trend_time_series
            + cosine_time_series
            + increased_noise_time_series
        )

        # Save the time series and the individual components as attributes
        return (
            final_synthetic_time_series,
            synthetic_time_series,
            trend_time_series,
            cosine_time_series,
            increased_noise_time_series,
        )

    def plot_time_series(self) -> None:
        """
        Plot the time series and its individual components.
        """
        fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(10, 20))
        ax[0].plot(self.synthetic_time_series)
        ax[0].set_title("Synthetic Time Series")
        ax[1].plot(self.base_time_series)
        ax[1].set_title("Base Time Series Component")
        ax[2].plot(self.trend_time_series)
        ax[2].set_title("Trend Event Time Series Component")
        ax[3].plot(self.cosine_time_series)
        ax[3].set_title("Cosine Event Time Series Component")
        ax[4].plot(self.increased_noise_time_series)
        ax[4].set_title("Increased Noise Event Time Series Component")
        plt.show()
        plt.close()
