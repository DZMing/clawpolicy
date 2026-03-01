#!/usr/bin/env python3
"""
Distributed training support - Redis + Celery

Realize multi-project parallel training function：
- RedisAs a message queue and state store
- CeleryPerform distributed task scheduling
- Support manyworkerParallel training
- Intelligent detection of dependencies（Downgrade to standalone mode when unavailable）

Phase 3.1 - expected200OK
"""

import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime
import logging

# try to importRedisandCelery
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore[assignment]

try:
    from celery import Celery, current_task
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    Celery = None  # type: ignore[assignment]
    current_task = None  # type: ignore[assignment]
    AsyncResult = None  # type: ignore[assignment]

from .trainer import RLTrainer


logger = logging.getLogger(__name__)


class DistributedTrainingConfig:
    """Distributed training configuration"""

    def __init__(self,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 broker_url: str | None = None,
                 result_backend: str | None = None,
                 require_worker: bool = True,
                 connection_timeout: float = 1.0):
        """
        Initial configuration

        Args:
            redis_host: RedisHost
            redis_port: Redisport
            redis_db: Redisdatabase
            broker_url: Celery broker URL
            result_backend: Celeryresults backendURL
            require_worker: Whether to require at least oneCelery workeronline
            connection_timeout: Connection detection timeout（Second）
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.require_worker = require_worker
        self.connection_timeout = connection_timeout

        # Used by defaultRedisasbrokerandbackend
        self.broker_url = broker_url or f"redis://{redis_host}:{redis_port}/{redis_db}"
        self.result_backend = result_backend or f"redis://{redis_host}:{redis_port}/{redis_db}"

        # Check availability
        self.redis_available = REDIS_AVAILABLE
        self.celery_available = CELERY_AVAILABLE


class DistributedTrainer:
    """
    Distributed trainer

    Function：
    - Multi-project parallel training
    - Task status tracking
    - Aggregation of training results
    - Smart downgrade（Use stand-alone mode when dependencies are unavailable）
    """

    def __init__(self, config: DistributedTrainingConfig | None = None,
                 model_dir: str | Path | None = None):
        """
        Initialize the distributed trainer

        Args:
            config: Distributed configuration
            model_dir: Model save directory
        """
        self.config = config or DistributedTrainingConfig()
        self.model_dir = Path(model_dir).expanduser() if model_dir else Path("./models/distributed")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Check if distributed functionality is available
        self.distributed_enabled = (
            self.config.redis_available and
            self.config.celery_available
        )

        if self.distributed_enabled and not self._distributed_runtime_ready():
            logger.warning("⚠️ No available distributed runtime detected，Downgrade to standalone mode")
            self.distributed_enabled = False

        if self.distributed_enabled:
            logger.info("✅ Distributed training is enabled（Redis + Celery）")
            self._initialize_distributed()
        else:
            logger.warning("⚠️ Distributed mode is not available，Downgrade to standalone mode")
            self._initialize_fallback()

        # Training task records
        self.training_tasks: Dict[str, Dict[str, Any]] = {}

    def _distributed_runtime_ready(self) -> bool:
        """Check if distributed runtime is available（Redisconnected + Optionalworkeronline）"""
        try:
            redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=True,
                socket_connect_timeout=self.config.connection_timeout,
                socket_timeout=self.config.connection_timeout,
            )
            redis_client.ping()
        except Exception as exc:
            logger.warning(f"⚠️ RedisConnection not available: {exc}")
            return False

        if not self.config.require_worker:
            return True

        try:
            probe_app = Celery(
                'openclaw_alignment_probe',
                broker=self.config.broker_url,
                backend=self.config.result_backend
            )
            replies = probe_app.control.ping(timeout=self.config.connection_timeout)
            if not replies:
                logger.warning("⚠️ Online not detectedCelery worker")
                return False
        except Exception as exc:
            logger.warning(f"⚠️ Celery workerDetection failed: {exc}")
            return False

        return True

    def _initialize_distributed(self):
        """Initialize distributed components"""
        # initializationRedisconnect
        self.redis_client = redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            decode_responses=True,
            socket_connect_timeout=self.config.connection_timeout,
            socket_timeout=self.config.connection_timeout,
        )

        # initializationCeleryapplication
        self.celery_app = Celery(
            'openclaw_alignment',
            broker=self.config.broker_url,
            backend=self.config.result_backend
        )

        # ConfigurationCelery
        self.celery_app.conf.update(
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
        )

        # Register training tasks
        self.celery_app.task(self._train_task, name='train_episode')

    def _initialize_fallback(self):
        """Initialize degraded mode（Standalone）"""
        self.local_trainer = RLTrainer(model_dir=str(self.model_dir))

    def train_distributed(self,
                        project_configs: List[Dict[str, Any]],
                        num_episodes_per_project: int = 100,
                        save_interval: int = 10) -> Dict[str, Any]:
        """
        Distributed training for multiple projects

        Args:
            project_configs: Project configuration list（Independent training for each project）
                [{"project_id": "proj1", "task_types": ["T1", "T2"]}, ...]
            num_episodes_per_project: Training for each projectepisodenumber
            save_interval: save interval

        Returns:
            Training statistics summary
        """
        logger.info(f"🚀 Start distributed training（{len(project_configs)} items）...")

        if not self.distributed_enabled:
            # Downgrade to stand-alone sequential training
            return self._train_sequential(project_configs, num_episodes_per_project)

        # Parallel training
        task_ids: List[Dict[str, str]] = []
        try:
            for config in project_configs:
                project_id = config.get("project_id", f"project_{len(task_ids)}")

                # Submit an asynchronous task
                result = self.celery_app.send_task(
                    'train_episode',
                    args=[config, num_episodes_per_project, save_interval],
                    kwargs={}
                )

                task_ids.append({
                    "project_id": project_id,
                    "task_id": result.id
                })

                # Record tasks
                self.training_tasks[project_id] = {
                    "task_id": result.id,
                    "status": "PENDING",
                    "config": config,
                    "started_at": datetime.now().isoformat()
                }
        except Exception as exc:
            logger.error(f"❌ Distributed task submission failed，Downgrade to standalone mode: {exc}")
            return self._train_sequential(project_configs, num_episodes_per_project)

        # Wait for all tasks to complete
        results = self._wait_for_tasks(task_ids)

        # Aggregation results
        return self._aggregate_results(results)

    def _train_sequential(self,
                         project_configs: List[Dict[str, Any]],
                         num_episodes_per_project: int) -> Dict[str, Any]:
        """
        sequential training（downgrade mode）

        Args:
            project_configs: Project configuration list
            num_episodes_per_project: Training for each projectepisodenumber

        Returns:
            Training statistics summary
        """
        logger.info("⚠️ Use stand-alone sequential training mode")

        all_results = []

        for config in project_configs:
            project_id = config.get("project_id", "unknown")

            logger.info(f"🔄 training items: {project_id}")

            # Create a standalone trainer
            trainer = RLTrainer(model_dir=str(self.model_dir / project_id))

            # train
            stats = trainer.train(
                num_episodes=num_episodes_per_project,
                max_steps_per_episode=100,
                save_interval=10
            )

            all_results.append({
                "project_id": project_id,
                "stats": stats
            })

        return self._aggregate_results(all_results)

    def _train_task(self,
                   project_config: Dict[str, Any],
                   num_episodes: int,
                   save_interval: int) -> Dict[str, Any]:
        """
        Celerytraining tasks（existworkerexecute on）

        Args:
            project_config: Project configuration
            num_episodes: trainepisodenumber
            save_interval: save interval

        Returns:
            training statistics
        """
        project_id = project_config.get("project_id", "unknown")

        # Update task status
        if CELERY_AVAILABLE and current_task:
            current_task.update_state(
                state='PROGRESS',
                meta={'project_id': project_id, 'progress': 0}
            )

        # Create a trainer
        model_dir = Path("./models/distributed") / project_id
        model_dir.mkdir(parents=True, exist_ok=True)

        trainer = RLTrainer(model_dir=str(model_dir))

        # train
        stats = trainer.train(
            num_episodes=num_episodes,
            max_steps_per_episode=100,
            save_interval=save_interval
        )

        return {
            "project_id": project_id,
            "stats": stats
        }

    def _wait_for_tasks(self,
                       task_ids: List[Dict[str, str]],
                       timeout: int = 3600) -> List[Dict[str, Any]]:
        """
        Wait for all tasks to complete

        Args:
            task_ids: TaskIDlist
            timeout: timeout（Second）

        Returns:
            Task result list
        """
        import time
        start_time = time.time()

        results: List[Dict[str, Any]] = []
        completed_count = 0

        while completed_count < len(task_ids):
            # Check timeout
            if time.time() - start_time > timeout:
                logger.error(f"❌ Training timeout（{timeout}Second）")
                break

            # Check the status of each task
            for task_info in task_ids:
                project_id = task_info["project_id"]
                task_id = task_info["task_id"]

                if project_id in [r["project_id"] for r in results]:
                    continue  # Completed

                # Query task status
                result = AsyncResult(task_id, app=self.celery_app)

                if result.ready():
                    # Mission accomplished
                    task_result = result.get()
                    results.append(task_result)

                    # Update record
                    self.training_tasks[project_id]["status"] = "COMPLETED"
                    self.training_tasks[project_id]["completed_at"] = datetime.now().isoformat()

                    completed_count += 1
                    logger.info(f"✅ project {project_id} Finish（{completed_count}/{len(task_ids)}）")

                elif result.status == 'FAILED':
                    # Task failed
                    logger.error(f"❌ project {project_id} Training failed: {result.info}")

                    self.training_tasks[project_id]["status"] = "FAILED"
                    self.training_tasks[project_id]["error"] = str(result.info)

                    completed_count += 1

            # Wait for some time and check again
            time.sleep(5)

        return results

    def _aggregate_results(self,
                          results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate training results

        Args:
            results: Training result list

        Returns:
            aggregate statistics
        """
        if not results:
            return {}

        # aggregated metrics
        total_rewards = [r["stats"].get("average_reward", 0) for r in results]
        total_episodes = sum(r["stats"].get("total_episodes", 0) for r in results)

        return {
            "total_projects": len(results),
            "total_episodes": total_episodes,
            "average_reward_per_project": {
                r["project_id"]: r["stats"].get("average_reward", 0)
                for r in results
            },
            "overall_average_reward": float(np.mean(total_rewards)),
            "reward_std": float(np.std(total_rewards)),
            "best_project": max(results, key=lambda x: x["stats"].get("average_reward", 0))["project_id"],
            "training_tasks": self.training_tasks
        }

    def get_task_status(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status

        Args:
            project_id: projectID

        Returns:
            Task status information
        """
        return self.training_tasks.get(project_id)

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all task status

        Returns:
            All task status
        """
        return self.training_tasks

    def cancel_task(self, project_id: str) -> bool:
        """
        Cancel task

        Args:
            project_id: projectID

        Returns:
            Is the cancellation successful?
        """
        if not self.distributed_enabled:
            logger.warning("⚠️ Standalone mode does not support task cancellation")
            return False

        task_info = self.training_tasks.get(project_id)
        if not task_info:
            logger.warning(f"⚠️ Task does not exist: {project_id}")
            return False

        task_id = task_info["task_id"]

        # CancelCeleryTask
        self.celery_app.control.revoke(task_id, terminate=True)

        # update status
        self.training_tasks[project_id]["status"] = "CANCELLED"
        self.training_tasks[project_id]["cancelled_at"] = datetime.now().isoformat()

        logger.info(f"✅ Task canceled: {project_id}")
        return True

    def save_training_report(self, results: Dict[str, Any], filename: str | None = None) -> str:
        """
        Save training report

        Args:
            results: Training results
            filename: file name

        Returns:
            report file path
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"distributed_training_report_{timestamp}.json"

        report_path = self.model_dir / filename

        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"✅ Training report saved: {report_path}")
        return str(report_path)


def main():
    """Test the distributed trainer"""
    config = DistributedTrainingConfig()
    trainer = DistributedTrainer(config)

    print("✅ Distributed trainer has been created")
    print(f"   distributed functions: {'✅ enable' if trainer.distributed_enabled else '❌ Disable（Downgrade）'}")

    # Test project configuration
    project_configs = [
        {"project_id": "test_project_1", "task_types": ["T1", "T2"]},
        {"project_id": "test_project_2", "task_types": ["T3", "T4"]},
    ]

    # train（A small amountepisodetest）
    results = trainer.train_distributed(
        project_configs=project_configs,
        num_episodes_per_project=3,
        save_interval=1
    )

    print("\n📊 Distributed training results:")
    print(f"   Total number of items: {results.get('total_projects', 0)}")
    print(f"   totalepisodenumber: {results.get('total_episodes', 0)}")
    print(f"   average reward: {results.get('overall_average_reward', 0):.3f}")

    # save report
    report_path = trainer.save_training_report(results)
    print(f"   training report: {report_path}")


if __name__ == "__main__":
    main()
