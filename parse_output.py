"""
Sample Output:

BEGINNING (1681841722.2563047): Baseline LR(0.0006) Heads(1) Embeddings(64) Block Size(64) Batch Size(32) Layers(2)
torch.Size([2048, 86])
tensor(4.6715, device='cuda:0', grad_fn=<NllLossBackward0>)
step 0: train loss 4.6570, val loss 4.6612 [1.0097455978393555 sec]
step 100: train loss 2.8579, val loss 2.8993 [2.566605806350708 sec]
step 200: train loss 2.6550, val loss 2.7157 [4.128730297088623 sec]
step 300: train loss 2.5632, val loss 2.6385 [5.6904895305633545 sec]
step 400: train loss 2.4776, val loss 2.5578 [7.243569850921631 sec]
2.402296543121338
Total Training Time: 7.827090263366699 seconds

bwaca A�ediusth sos owond anis oneI br pntcowly
r ac ne vant sism fsat, wa ay 5g fovaemo s, Th lkini
BEGINNING (1681841730.2903962): Baseline LR(0.0006) Heads(1) Embeddings(64) Block Size(64) Batch Size(32) Layers(4)
"""

from enum import Enum, auto


class OutputState(Enum):
    Unknown = auto()
    Beginning = auto()
    Size = auto()
    Tensor = auto()
    Step = auto()
    FinalLoss = auto()
    TotalTrainingTime = auto()
    GeneratedText = auto()


class ExperimentInfo:

    def __init__(self, text_to_parse: str):
        self.steps = []
        self.generated_text = []
        info = text_to_parse.split('Baseline ')[1].strip().rstrip(')')
        for field in info.split(') '):
            name, value = field.split('(')
            self.__dict__[name] = value

    def get_headers(self) -> list[str]:
        headers = [header for header in self.__dict__.keys() if header not in['steps', 'generated_text']]
        headers += [f"Step {s.step} Loss" for s in self.steps]
        headers.append("Generated Text")
        return headers

    def to_list(self) -> list:
        results = [value.strip() for header, value in self.__dict__.items() if header not in['steps', 'generated_text']]
        results += [s.training_loss for s in self.steps]
        results.append(str.join("\\n", [s.strip() for s in self.generated_text]))
        return results

    def to_step_rows(self) -> list:
        base_line_row = [value.strip() for header, value in self.__dict__.items()
                         if header not in['steps', 'generated_text', "final_loss", "total_training_time", "Size"]]
        return [base_line_row + [step.step, step.training_loss, step.validation_loss,
                                 step.accumulated_time_sec]
                for step in self.steps]

    def to_step_headers(self) -> list:
        base_line_row = [header.strip() for header, value in self.__dict__.items()
                         if header not in['steps', 'generated_text', "final_loss", "total_training_time", "Size"]]
        return base_line_row + ["Step", "Training Loss", "Validation Loss", "Accumulated Time Sec"]


class StepInfo:

    def __init__(self, text_to_parse: str):
        step, values = text_to_parse.split(':')
        self.step = step.replace('step ', '')
        values = values.replace('[', ',') # since split only works on one char
        parts = values.split(',')
        self.training_loss = parts[0].split(' ')[-1]
        self.validation_loss = parts[1].strip().split(' ')[-1]
        self.accumulated_time_sec = parts[2].split(' ')[0]

class ProcessOutput:

    def __init__(self):
        self.current_state: OutputState
        self.current_info: ExperimentInfo
        self.experiments = []

    def process_output(self, file_name: str):

        with open(file_name, "r") as f:
            self.current_state = OutputState.Unknown
            self.current_info: ExperimentInfo = None
            for line in f:
                if self.current_state == OutputState.Unknown:
                    self._handle_beginning(line)
                elif self.current_state == OutputState.Beginning:
                    self._handle_size(line)
                elif self.current_state == OutputState.Size:
                    self._handle_tensor(line)
                elif self.current_state == OutputState.Tensor:
                    self._handle_step(line)
                elif "Step" in self.current_state.name:
                    self._handle_step(line)
                elif self.current_state == OutputState.FinalLoss:
                    self._handle_total_training_time(line)
                elif self.current_state == OutputState.TotalTrainingTime:
                    self.current_state = OutputState.GeneratedText
                    # do nothing else, should be a blank line
                elif self.current_state == OutputState.GeneratedText:
                    if "BEGINNING" in line:
                        self._handle_beginning(line)
                        continue
                    self.current_info.generated_text.append(line)


    def _handle_beginning(self, line):
        if "BEGINNING" not in line:
            return
        self.current_state = OutputState.Beginning
        self.current_info = ExperimentInfo(line)
        self.experiments.append(self.current_info)

    def _handle_size(self, line):
        if not line.startswith("torch.Size"):
            return
        self.current_info.Size = line.replace("torch.Size(", "").replace(")", "")
        self.current_state = OutputState.Size

    def _handle_tensor(self, line):
        if not line.startswith("tensor("):
            return
        self.current_state = OutputState.Tensor

    def _handle_step(self, line):
        if not line.startswith("step "):
            self._handle_final_loss(line)
            return
        step = line.split(':')[0].title()
        step = step.replace(' ', '')
        self.current_state = OutputState.Step
        self.current_info.steps.append(StepInfo(line))

    def _handle_final_loss(self, line):
        self.current_state = OutputState.FinalLoss
        self.current_info.final_loss = line

    def _handle_total_training_time(self, line):
        if not line.startswith("Total Training Time:"):
            return
        self.current_state = OutputState.TotalTrainingTime
        self.current_info.total_training_time = line.replace("Total Training Time:", "").replace("seconds", "").strip()

    def save_to_csv(self, file_name: str):
        if not self.experiments:
            print("No experiments to process!")
            return

        headers = self.current_info.get_headers()
        from csv import writer
        with open(file_name, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = writer(csv_file)
            csv_writer.writerow(headers)
            for experiment in self.experiments:
                csv_writer.writerow(experiment.to_list())

    def save_steps_to_csv(self, file_name: str):
        if not self.experiments:
            print("No experiments to process!")
            return

        headers = self.current_info.to_step_headers()
        from csv import writer
        with open(file_name, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = writer(csv_file)
            csv_writer.writerow(headers)
            for experiment in self.experiments:
                csv_writer.writerows(experiment.to_step_rows())


if __name__ == '__main__':
    from pathlib import Path
    curdir = Path(".")
    for file_name in curdir.glob("**/output.txt"):
        print(file_name)
        po = ProcessOutput()
        po.process_output(file_name)
        po.save_to_csv(str(file_name).replace(".txt",".csv"))
        po.save_steps_to_csv(str(file_name).replace(".txt", "_steps.csv"))