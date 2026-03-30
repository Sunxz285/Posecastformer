import os
from tensorboard.backend.event_processing import event_accumulator


def read_tfevents(file_path):
    """
    Parse and print scalar content from TensorBoard event files
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found {file_path}")
        return

    try:
        # Initialize EventAccumulator to load data
        # size_guidance specifies loading all data to avoid truncation due to large data volume
        ea = event_accumulator.EventAccumulator(
            file_path,
            size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()

        # Get all available scalar tags (e.g., loss, accuracy, etc.)
        tags = ea.Tags()['scalars']

        if not tags:
            print("No scalar data found in this file (Scalars).")
            print("Other data keys included:", ea.Tags().keys())
            return

        print(f"Found the following tags: {tags}\n")

        # Iterate through each tag and print corresponding data points
        for tag in tags:
            print(f"--- Tag: {tag} ---")
            events = ea.Scalars(tag)
            for event in events:
                # event.step: training step
                # event.value: recorded value
                # event.wall_time: timestamp
                print(f"Step: {event.step:5d} | Value: {event.value:.6f} | Time: {event.wall_time}")
            print("\n")

    except Exception as e:
        print(f"Error occurred while parsing file: {e}")


if __name__ == "__main__":
    # Filename
    target_file = ('checkpoint/pose3d/posecastformer_cpn/logs/events.out.tfevents.1772845304.autodl-container-3d94448d17-9c82ef61'
                   )

    # Execute reading
    read_tfevents(target_file)